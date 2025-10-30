"""
Author: Akram Moustafa
Date: 10/20/2025
Purpose:
  Two-channel Navier–Stokes dataset loader for vorticity + temperature.
  Uses local .pt files (no download).
"""

from pathlib import Path
from torch.utils.data import DataLoader

from .pt_dataset import PTDataset
from neuralop.utils import get_project_root

class NavierStokes2ChDataset(PTDataset):
    def __init__(self,
                 root_dir: Path,
                 n_train: int,
                 n_tests: list,
                 batch_size: int,
                 test_batch_sizes: list,
                 train_resolution: int = 128,
                 test_resolutions: list = [128,128],
                 encode_input: bool = True,
                 encode_output: bool = True,
                 encoding: str = "channel-wise",
                 channel_dim: int = 2,
                 subsampling_rate=None,
                 download: bool = False,
                 channels_squeezed: bool = False):
        """
        Loads pre-saved 2-channel Navier–Stokes data (.pt) from disk.
        Expected file names:
            ns2ch_train_<res>.pt
            ns2ch_test_<res>.pt
        """

        root_dir = Path(root_dir)
        assert root_dir.exists(), f"Data directory not found: {root_dir}"

        dataset_name = "ns2ch"  # prefix for your .pt files

        super().__init__(root_dir=root_dir,
                         n_train=n_train,
                         n_tests=n_tests,
                         dataset_name=dataset_name,
                         batch_size=batch_size,
                         test_batch_sizes=test_batch_sizes,
                         train_resolution=train_resolution,
                         test_resolutions=test_resolutions,
                         encode_input=encode_input,
                         encode_output=encode_output,
                         encoding=encoding,
                         channel_dim=channel_dim,
                         input_subsampling_rate=subsampling_rate,
                         output_subsampling_rate=subsampling_rate,
                         channels_squeezed = channels_squeezed)
        
def load_navier_stokes_2ch_pt(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    data_root=None,
    train_resolution=128,
    test_resolutions=[128,128],
    encode_input=True,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=2,
    subsampling_rate=None,
    num_workers=1,
    channels_squeezed = False
):
    """
    Returns train/test DataLoaders and DataProcessor for 2-channel dataset.
    """
    if data_root is None:
        data_root=Path("C:/Users/ammou/Documents/neuraloperator/datasets/data")

    dataset = NavierStokes2ChDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        channel_dim=channel_dim,
        subsampling_rate=subsampling_rate,
        channels_squeezed = channels_squeezed
    )

    train_loader = DataLoader(dataset.train_db,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loaders = {
        res: DataLoader(dataset.test_dbs[res],
                        batch_size=test_batch_sizes[0],
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True)
        for res in test_resolutions
    }

    return train_loader, test_loaders, dataset.data_processor
