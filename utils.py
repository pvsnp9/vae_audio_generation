import librosa
import os
import torchaudio
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hyperparams import HyperParams
from spectrogramdataset import SpectrogramDataset
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
from raw_dataset import AudioDataset 

def get_audio_metadata(file_path:str)->tuple:
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        channels = y.shape[0] if y.ndim > 1 else 1
        duration = librosa.get_duration(y=y, sr=sr)
        return sr, channels, duration
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


def convert_audio_to_spectrogram(root_dir:str, save_dir:str, sample_rate:int = 22050, 
                                 duration=30, n_fft:int = 1024, hop_length:int = 512, n_mels: int = 128 ):
    """
    Converts audio files in root_dir organized as root_dir/<genre>/*.wav into spectrogram images.
    Each audio file is resampled to `sample_rate`, trimmed/padded to `duration` seconds, and converted
    to a Mel spectrogram in decibel scale. The resulting images are saved in save_dir/<genre>/ with the same
    file name (but with .png extension) and with exact pixel dimensions matching the spectrogram array.

    This version includes try–except blocks to handle errors during file loading and processing.

    Parameters:
      root_dir (str): Directory containing genre folders with audio files.
      save_dir (str): Directory where spectrogram images will be saved.
      sample_rate (int): Desired sample rate (e.g., 22050).
      duration (int or float): Duration in seconds to trim or pad each audio file.
    """
    # Define transforms for computing the Mel spectrogram and converting to dB.
    mel_spec_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,       # FFT window size
        hop_length=hop_length,   # Hop length between frames
        n_mels=n_mels        # Number of Mel bands
    )
    db_transform = T.AmplitudeToDB()
    
    # Calculate the target number of samples for the given duration.
    target_samples = int(sample_rate * duration)
    
    # Create the save directory if it doesn't exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Iterate through each genre folder in root_dir.
    genres = os.listdir(root_dir)
    for genre in genres:
        genre_path = os.path.join(root_dir, genre)
        if os.path.isdir(genre_path):
            # Create corresponding genre folder in save_dir.
            genre_save_dir = os.path.join(save_dir, genre)
            if not os.path.exists(genre_save_dir):
                os.makedirs(genre_save_dir)
            
            # Process each .wav file in the genre folder.
            for file_name in os.listdir(genre_path):
                if file_name.lower().endswith('.wav'):
                    file_path = os.path.join(genre_path, file_name)
                    try:
                        # Try loading the audio file.
                        waveform, sr = torchaudio.load(file_path)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue

                    try:
                        # Resample if needed.
                        if sr != sample_rate:
                            resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
                            waveform = resampler(waveform)
                        
                        # Convert to mono if necessary.
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        
                        # Trim or pad the waveform to the target length.
                        if waveform.shape[1] < target_samples:
                            pad_amount = target_samples - waveform.shape[1]
                            waveform = F.pad(waveform, (0, pad_amount))
                        elif waveform.shape[1] > target_samples:
                            waveform = waveform[:, :target_samples]
                        
                        # Compute the Mel spectrogram and convert to decibel scale.
                        mel_spec = mel_spec_transform(waveform)
                        mel_spec_db = db_transform(mel_spec)
                        mel_spec_np = mel_spec_db.squeeze().numpy()  # Shape: [n_mels, time_frames]
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue
                    
                    try:
                        # Save the spectrogram image with exact pixel dimensions.
                        save_file = os.path.join(genre_save_dir, os.path.splitext(file_name)[0] + '.png')
                        plt.imsave(save_file, mel_spec_np, origin='lower', cmap='gray')
                        # print(f"Saved spectrogram for {file_path} to {save_file}")
                    except Exception as e:
                        print(f"Error saving spectrogram for {file_path}: {e}")
                        continue

        print(f"Saved spectrogram for {genre} to {genre_save_dir}")


def get_spectrogram_dataloader(hyper_params: HyperParams)->tuple:
    """
    Creates a DataLoader for spectrogram images.

    AArgs:
            hyper_params.spectrogram_data_dir (str): Root directory containing spectrogram images in subfolders.
            hyper_params.image_size (tuple): Desired image size (H, W) for resizing.
            hyper_params.batch_size (int): Batch size for the DataLoader
    
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = SpectrogramDataset(hyper_params=hyper_params)

    train_size = len(dataset) - int(len(dataset) * hyper_params.validation_size)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=hyper_params.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyper_params.batch_size, shuffle=True)
    
    return train_loader, val_loader


def plot_generated_audio(audio: torch.Tensor, sample_rate: int = 3000) -> None:
    """
    Plots generated audio samples.

    Args:
        audio (torch.Tensor): The generated audio tensor. It can have one of these shapes:
                              - (length,) for a single-channel, single-sample signal,
                              - (channels, length) for a multi-channel single-sample signal (plots first channel),
                              - (num_samples, channels, length) for a batch of samples (plots first sample, first channel).
        sample_rate (int): Sample rate of the audio used to create the time axis.
    """
    # Handle different tensor dimensions.
    if audio.ndim == 3:
        # Audio has shape (num_samples, channels, length).
        num_samples, channels, length = audio.shape
        # For demonstration, we plot the first sample's first channel.
        waveform = audio[0, 0].cpu().numpy()
        title = "Generated Audio (Sample 0, Channel 0)"
    elif audio.ndim == 2:
        # Audio has shape (channels, length) for a single sample.
        channels, length = audio.shape
        waveform = audio[0].cpu().numpy()  # Plot the first channel.
        title = "Generated Audio (Channel 0)"
    elif audio.ndim == 1:
        # Audio has shape (length,)
        length = audio.shape[0]
        waveform = audio.cpu().numpy()
        title = "Generated Audio"
    else:
        raise ValueError("Audio tensor has an unsupported number of dimensions.")
    
    # Create a time axis.
    time_axis = np.linspace(0, length / sample_rate, num=length)
    
    # Plot the waveform.
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, waveform)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def vae_loss(x: torch.Tensor,  x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta:float = 1.0) -> Tuple[torch.Tensor, ...]:
    # MSE
    mse = F.mse_loss(x_recon, x, reduction='mean')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return mse + beta*kl, mse, kl


def compute_audio_stats(dataset: AudioDataset) -> Tuple[float, float]:
    all_waveforms = []
    
    # 1. Collect all waveforms
    for waveform, _ in dataset:  # (channels, samples)
        all_waveforms.append(waveform)
    
    # 2. Stack into single tensor
    stacked = torch.cat(all_waveforms, dim=1)  # (channels, all_samples)
    
    # 3. Compute statistics
    mean = torch.mean(stacked).item()
    std = torch.std(stacked).item()
    
    return mean, std