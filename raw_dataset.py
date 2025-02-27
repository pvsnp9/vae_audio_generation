from typing import Callable, Optional, Tuple, List
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import soundfile as sf
from hyperparams import RawHyperParams

class AudioDataset(Dataset):
    def __init__(self, raw_hp: RawHyperParams) -> None:
        """
        Args:
            raw_hp (RawHyperParams): Hyperparameters containing directory, sampling rate, etc.
            transform (Optional[Callable]): Optional transform to be applied on the waveform.
        """
        self.data_dir: str = raw_hp.data_genres_original_dir
        self.sample_rate: float = raw_hp.sampling_rate
        self.duration: float = raw_hp.duration
        self.offset: float = raw_hp.offset
        self.num_samples: int = int(raw_hp.sampling_rate * raw_hp.duration)
        
        # Mean and std for standardization (assumed to be provided in raw_hp)
        self.mean: float = raw_hp.mean
        self.std: float = raw_hp.std
        
        self.audio_files: List[Tuple[str, str]] = []  # List of tuples: (audio_file_path, genre)

        # Walk through the directory and subdirectories to get all .wav files.
        # Using soundfile (sf) to verify file validity.
        for root, dirs, files in os.walk(self.data_dir):
            genre: str = os.path.basename(root)
            for file in files:
                if file.endswith('.wav'):
                    file_path: str = os.path.join(root, file)
                    try:
                        _ = sf.info(file_path)
                        self.audio_files.append((file_path, genre))
                    except Exception as e:
                        print(f"Skipping corrupt file: {file_path} - {e}")

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        audio_path, genre = self.audio_files[idx]
        try:
            # Load the audio with librosa.
            # Setting mono=False to preserve channels.
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, offset=self.offset, duration=self.duration, mono=False)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Create a zero waveform if loading fails.
            waveform = np.zeros((1, self.num_samples))
            sr = self.sample_rate

        # If the audio is mono, librosa.load returns a 1D array.
        # Add a channel dimension so that waveform shape becomes (channels, samples)
        if waveform.ndim == 1:
            waveform = np.expand_dims(waveform, axis=0)

        # Convert the numpy array to a torch tensor.
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # Ensure the waveform has exactly self.num_samples samples.
        if waveform.shape[1] < self.num_samples:
            pad_amount = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]

        # Standardize the data using the given mean and std.
        # waveform = (waveform - self.mean) / self.std

        return waveform, genre
