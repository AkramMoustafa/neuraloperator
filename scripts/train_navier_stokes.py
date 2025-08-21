import sys

from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler, random_split
import torch.distributed as dist
import wandb
from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.mpu.comm import get_local_rank
from neuralop.training import setup, AdamW

config_name = "default"
from zencfg import make_config_from_cli
import sys 
sys.path.insert(0, '../')
from config.navier_stokes_config import Default_NS2D_2ch as Cfg
config = make_config_from_cli(Cfg)
config = config.to_dict()

class ChannelwiseGaussianNormalizer:
    def __init__(self, sample_loader, key, eps=1e-6, max_batches=None):
        sums = sqs = None; n = 0
        with torch.no_grad():
            for bi, batch in enumerate(sample_loader):
                x = batch[key].float()  # [B,C,H,W]
                B, C = x.shape[:2]
                x = x.view(B, C, -1)
                sums = x.sum((0,2)) if sums is None else sums + x.sum((0,2))
                sqs  = (x**2).sum((0,2)) if sqs is None else sqs + (x**2).sum((0,2))
                n += x.shape[0] * x.shape[2]
                if max_batches and (bi+1) >= max_batches: break
        self.mean = (sums / n).view(1,-1,1,1)
        var = (sqs / n).view(1,-1,1,1) - self.mean**2
        self.std  = var.clamp_min(0.).sqrt() + eps

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
        return self

    def encode(self, x): return (x - self.mean) / self.std
    def decode(self, x): return x * self.std + self.mean

class SimpleProcessor:
    def __init__(self, inn, outn):
        self.inn = inn
        self.outn = outn
        # expose these names for compatibility with patching wrappers
        self.in_normalizer  = inn
        self.out_normalizer = outn
        self._training = True
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        self.inn.to(device)
        self.outn.to(device)
        return self

    def train(self):
        self._training = True
        return self

    def eval(self):
        self._training = False
        return self

    def preprocess(self, sample):
        """
        Expect sample = {'x': Tensor, 'y': Tensor}
        Move to device and normalize (encode).
        """
        x = sample["x"]
        y = sample["y"]
        if isinstance(x, torch.Tensor):
            x = x.to(self.device, non_blocking=True)
        if isinstance(y, torch.Tensor):
            y = y.to(self.device, non_blocking=True)
        x = self.inn.encode(x)
        y = self.outn.encode(y)
        return {"x": x, "y": y}

    def postprocess(self, out, sample=None):
        """
        Decode model outputs (and optionally the original sample).
        Return (decoded_out, decoded_sample) as expected by Trainer.
        """
        decoded_out = {}

        if isinstance(out, torch.Tensor):
            decoded_out["y"] = self.outn.decode(out).detach().cpu()
        elif isinstance(out, dict):
            if "y" in out:
                decoded_out["y"] = self.outn.decode(out["y"]).detach().cpu()
            elif "output" in out:
                decoded_out["y"] = self.outn.decode(out["output"]).detach().cpu()
            else:

                for v in out.values():
                    if isinstance(v, torch.Tensor):
                        decoded_out["y"] = self.outn.decode(v).detach().cpu()
                        break

        decoded_sample = None
        if sample is not None:
            decoded_sample = {}
            if "x" in sample and isinstance(sample["x"], torch.Tensor):
                decoded_sample["x"] = self.inn.decode(sample["x"]).detach().cpu()
            if "y" in sample and isinstance(sample["y"], torch.Tensor):
                decoded_sample["y"] = self.outn.decode(sample["y"]).detach().cpu()

        return decoded_out, decoded_sample

    # Optional helpers if you call them elsewhere
    def encode(self, batch):
        return {"x": self.inn.encode(batch["x"]), "y": self.outn.encode(batch["y"])}

    def decode(self, batch):
        out = {}
        if "x" in batch: out["x"] = self.inn.decode(batch["x"])
        if "y" in batch: out["y"] = self.outn.decode(batch["y"])
        return out

# --- custom function to create loaders + processor ---
def make_train_test_loaders(dataset, batch_size=4, test_fraction=0.2, num_workers=0):
    """Split dataset, build train/test loaders, and return normalizer data_processor."""
    n_total = len(dataset)
    n_test = max(1, int(test_fraction * n_total))
    n_train = n_total - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    stat_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    in_norm  = ChannelwiseGaussianNormalizer(stat_loader, key="x", max_batches=8)  # CHANGED to 'x'
    out_norm = ChannelwiseGaussianNormalizer(stat_loader, key="y", max_batches=8) # CHANGED to 'y'
    data_processor = SimpleProcessor(in_norm, out_norm)

    test_loaders = {64: test_loader}

    return train_loader, test_loaders, data_processor


T, H, W = 40, 64, 64
omega = torch.randn(T, 1, H, W)
temp  = torch.randn(T, 1, H, W)

class WT2Channel(torch.utils.data.Dataset):
    def __init__(self, omega, temp, dt=1):
        self.omega, self.temp, self.dt = omega, temp, dt
        self.T = omega.shape[0]
    def __len__(self): return self.T - self.dt
    def __getitem__(self, i):
        x = torch.cat([self.omega[i],        self.temp[i]],        dim=0)
        y = torch.cat([self.omega[i+self.dt], self.temp[i+self.dt]], dim=0)
        return {"x": x, "y": y}  # CHANGED to x/y

dataset = WT2Channel(omega, temp, dt=1)

train_loader, test_loaders, data_processor = make_train_test_loaders(
    dataset, batch_size=4, test_fraction=0.2
)

print("Train batches:", len(train_loader))
for batch in train_loader:
    print(batch["x"].shape, batch["y"].shape)  # CHANGED to x/y
    break


# Set-up distributed communication, if using
device, is_logger = setup(config)
# Set up WandB logging
wandb_init_args = None
if config.wandb.log and is_logger:
    print(config.wandb.log)
    print(config)
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.model.n_layers,
                config.model.n_modes,
                config.model.hidden_channels,
                config.model.factorization,
                config.model.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger
# Print config to screen
if config.verbose:
    print(f"##### CONFIG #####\n")
    print(config)

model = get_model(config)
model = model.to(device)

# convert dataprocessor to an MGPatchingDataprocessor if patching levels > 0
if config.patching.levels > 0:
    # NOTE: relies on .in_normalizer/.out_normalizer exposed above
    data_processor = MGPatchingDataProcessor(model=model,
                                             in_normalizer=data_processor.in_normalizer,
                                             out_normalizer=data_processor.out_normalizer,
                                             padding_fraction=config.patching.padding,
                                             stitching=config.patching.stitching,
                                             levels=config.patching.levels,
                                             use_distributed=config.distributed.use_distributed)
data_processor = data_processor.to(device)


if config.distributed.use_distributed:
    train_db = train_loader.dataset
    train_sampler = DistributedSampler(train_db, rank=get_local_rank())
    train_loader = DataLoader(dataset=train_db,
                              batch_size=config.data.batch_size,
                              sampler=train_sampler)
    for (res, loader), batch_size in zip(test_loaders.items(), config.data.test_batch_sizes):
        
        test_db = loader.dataset
        test_sampler = DistributedSampler(test_db, rank=get_local_rank())
        test_loaders[res] = DataLoader(dataset=test_db,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=test_sampler)

# Create the optimizer
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")

# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f'Got training_loss={config.opt.training_loss} '
        f'but expected one of ["l2", "h1"]'
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    mixed_precision=config.opt.mixed_precision,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log = config.wandb.log
)

if is_logger:
    n_params = count_model_params(model)
    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()
    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        wandb.watch(model)

trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if config.wandb.log and is_logger:
    wandb.finish()

if dist.is_initialized():
    dist.destroy_process_group()
