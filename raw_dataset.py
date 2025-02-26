from typing import Callable, Optional, Tuple, List
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from hyperparams import RawHyperParams

class AudioDataset(Dataset):
    def __init__(self, raw_hp: RawHyperParams, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
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
        self.transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = transform
        self.audio_files: List[Tuple[str, str]] = []  # List of tuples: (audio_file_path, genre)

        # Walk through the directory and subdirectories to get all .wav files.
        for root, dirs, files in os.walk(self.data_dir):
            genre: str = os.path.basename(root)
            for file in files:
                if file.endswith('.wav'):
                    file_path: str = os.path.join(root, file)
                    # Try to load metadata; if it fails, skip this file.
                    try:
                        _ = torchaudio.info(file_path)
                        self.audio_files.append((file_path, genre))
                    except Exception as e:
                        print(f"Skipping corrupt file: {file_path} - {e}")

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        audio_path, genre = self.audio_files[idx]
        waveform: Optional[torch.Tensor] = None

        try:
            info = torchaudio.info(audio_path)
            file_sr: float = info.sample_rate
            frame_offset: int = int(self.offset * file_sr)
            num_frames: int = int(self.duration * file_sr)
            waveform, sr = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)
        except Exception as e:
            # This should rarely happen because we filtered corrupt files in __init__.
            print(f"Error loading {audio_path}: {e}")
            waveform = torch.zeros(1, self.num_samples)
            sr = self.sample_rate

        if sr != self.sample_rate:
            try:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            except Exception as e:
                print(f"Error resampling {audio_path}: {e}")
                waveform = torch.zeros(1, self.num_samples)

        # Ensure the waveform has exactly self.num_samples samples.
        if waveform.shape[1] < self.num_samples:
            pad_amount = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, genre
