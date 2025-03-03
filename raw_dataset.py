from typing import Callable, Optional, Tuple, List
import os
import numpy as np
import torch
import os
import glob
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import soundfile as sf
from hyperparams import RawHyperParams, TVAEParams
import torch.nn.functional as F



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


"""TVAE dataset"""
class TVAESeqSeqDataset(Dataset):
    def __init__(self, hp:TVAEParams) -> None:
        self.target_sr = hp.sampling_rate
        self.seq_length = hp.seq_len
        self.step =  hp.window_step
        self.num_required_samples = int(hp.duration * hp.sampling_rate)  #10 sec * 8000 = 80000 samples
        
        self.data = []  # List of tuples: (waveform, genre, filename)
        self.index = []  # List of tuples: (data_index, start_sample)
        
        # Traverse genre folders
        try:
            genre_dirs = [d for d in os.listdir(hp.data_genres_original_dir) if os.path.isdir(os.path.join(hp.data_genres_original_dir, d))]
        except Exception as e:
            print(f"Error listing directories in {hp.data_genres_original_dir}: {e}")
            genre_dirs = []
            
        for genre in genre_dirs:
            genre_path = os.path.join(hp.data_genres_original_dir, genre)
            wav_files = glob.glob(os.path.join(genre_path, "*.wav"))
            for wav_file in wav_files:
                try:
                    waveform, sr = torchaudio.load(wav_file)  # waveform shape: [channels, samples]
                except Exception as e:
                    print(f"Error loading file {wav_file}: {e}")
                    continue
                
                # Resample if needed
                try:
                    if sr != self.target_sr:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=hp.sampling_rate)
                        waveform = resampler(waveform)
                except Exception as e:
                    print(f"Error resampling file {wav_file}: {e}")
                    continue
                    
                # Convert to mono if necessary
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                    
                # Skip if file is too short
                if waveform.shape[1] < self.num_required_samples:
                    continue
                
                # Truncate to the first 10 seconds
                waveform = waveform[:, :self.num_required_samples]
                data_idx: int = len(self.data)
                self.data.append((waveform.contiguous(), genre, os.path.basename(wav_file)))
                
                # Build sliding window index: ensure both src and tgt have full seq_length samples.
                for start in range(0, self.num_required_samples - self.seq_length, self.step):
                    self.index.append((data_idx, start))
        
        print(f"Loaded {len(self.data)} files with a total of {len(self.index)} sliding window segments.")

    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_idx, start = self.index[idx]
        waveform, genre, filename = self.data[data_idx]
        # Extract src: [start, start+seq_length]
        src = waveform[:, start:start + self.seq_length]
        # Extract tgt: [start+1, start+seq_length+1]
        tgt = waveform[:, start + 1:start + self.seq_length + 1]
        return src, tgt
