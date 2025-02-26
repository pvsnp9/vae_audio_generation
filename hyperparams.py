from dataclasses import dataclass
import torch

@dataclass
class HyperParams:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size: list[int] = (128,1290)
    batch_size: int = 128
    validation_size:float = 0.2
    num_blocks:int = 2
    latent_dim:int = 128
    input_size: list[int] = (128,1290)
    input_channels:int = 1

    lr: float = 1e-3
    num_epochs: int = 50

    block_type_encoder:str = "encoder"
    block_type_decoder:str = "decoder"

    # directories
    data_root_dir:str = "data"
    data_genres_original_dir:str = "data/genres_original"
    data_images_original_dir:str = "data/images_original"
    spectrogram_data_dir:str = "data/spectrograms"

    model_dir:str = "models"
    model_file_name:str = "resenet_vae.pth"
    output_dir:str = "outputs"

    output_audio_dir:str = "outputs/audio"

    log_dir:str = "log"
    train_log_file:str = "trainlog.json"


    # audio params
    sampling_rate:float = 22050
    duration: int = 30 # seconds
    # FFT window size
    n_fft:int = 1024
    # Hop length between frames
    hop_length:int = 512
    # Number of Mel bands
    n_mels:int = 128       


@dataclass
class RawHyperParams:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size: int = 32
    validation_size:float = 0.2
    num_blocks:int = 2
    latent_dim:int = 32
    input_size: list[int] = (90000,1)
    input_channels:int = 1
    input_length: float = input_size[0] * input_size[1]

    hidden_dims:list[int] =(16, 32, 64) #List of hidden channel dimensions for encoder blocks.

    lr: float = 1e-5
    num_epochs: int = 50

    block_type_encoder:str = "encoder"
    block_type_decoder:str = "decoder"

    # directories
    data_root_dir:str = "data"
    data_genres_original_dir:str = "data/genres_original"
    data_images_original_dir:str = "data/images_original"
    spectrogram_data_dir:str = "data/spectrograms"

    model_dir:str = "models"
    model_file_name:str = "resenet_vae.pth"
    output_dir:str = "outputs"

    output_audio_dir:str = "outputs/audio"

    log_dir:str = "log"
    train_log_file:str = "trainlog.json"


    # audio params
    sampling_rate:float = 3000
    duration: int = 30 # seconds
    offset:float = 0.0

