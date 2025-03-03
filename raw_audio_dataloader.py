from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from hyperparams import RawHyperParams, TVAEParams
from raw_dataset import AudioDataset, TVAESeqSeqDataset

class AudioDataLoader:
    def __init__(self, raw_hp: RawHyperParams, shuffle: bool = True, num_workers: int = 0) -> None:
        self.dataset = AudioDataset(raw_hp=raw_hp)
        self.loader = DataLoader(
            self.dataset,
            batch_size=raw_hp.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

def get_train_val_loaders(raw_hp: RawHyperParams, shuffle: bool = True, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    # Create the full dataset.
    
    dataset = AudioDataset(raw_hp=raw_hp)
    
    # Compute split sizes.
    val_size = int(raw_hp.validation_size * len(dataset))
    train_size = len(dataset) - val_size
    
    # Split dataset into training and validation.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=raw_hp.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=raw_hp.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    return train_loader, val_loader





"""TAVE data loader"""
def get_tvae_dataloaders(hp: TVAEParams, num_workers=1) -> Tuple[DataLoader, DataLoader]:
    """
    Creates training and validation DataLoaders from the audio dataset.

    Args:
        root_dir (str): Directory containing genre subdirectories.
        batch_size (int): Number of samples per batch.
        segment_length (int): Fixed length for each audio segment.
        val_split (float): Fraction of data to use for validation.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    dataset = TVAESeqSeqDataset(hp=hp)
    total_samples: int = len(dataset)
    val_size: int = int(total_samples * hp.validation_size)
    train_size: int = total_samples - val_size

    # Split dataset randomly
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    try:
        train_loader: DataLoader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=num_workers)
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=num_workers)
    except Exception as e:
        raise RuntimeError(f"Error creating DataLoaders: {e}")
    
    return train_loader, val_loader


# if __name__ == '__main__':
#     from hyperparams import RawHyperParams
#     raw_hp = RawHyperParams()
#     raw_hp.batch_size = 4 
    
#     train_loader, val_loader = get_train_val_loaders(raw_hp=raw_hp, shuffle=True, num_workers=0)
    
#     for batch_waveforms, batch_genres in train_loader:
#         print(f"Train batch waveform shape: {batch_waveforms.shape}")
#         print(f"Train batch genres: {batch_genres}")
#         break  
    
#     for batch_waveforms, batch_genres in val_loader:
#         print(f"Validation batch waveform shape: {batch_waveforms.shape}")
#         print(f"Validation batch genres: {batch_genres}")
#         break 


#       TVAE TEST
        # hp =TVAEParams()
        # hp.batch_size = 4
        # root_directory: str = "data/genres_original"  # Path to your dataset directory
        # try:
        #     train_loader, val_loader = get_dataloaders(hp=hp)
        # except Exception as e:
        #     print(e)
        
        # try:
        #     for batch_waveforms, batch_labels in train_loader:
        #         print("Batch waveforms shape:", batch_waveforms.shape)  # Expected: (batch, channels, segment_length)
        #         print("Batch labels:", batch_labels)
        #         break  # Remove break to iterate through full loader
        # except Exception as e:
        #     print(f"Error during iteration: {e}")