import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Union
from resenetblock import Resnet1dBlock
from hyperparams import RawHyperParams
from datetime import datetime

class ResenetVAE(nn.Module):
    def __init__(
        self,
        hyperparams: RawHyperParams,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        activation: nn.Module = nn.ReLU(inplace=True)
    ) -> None:
        super(ResenetVAE, self).__init__()
        self.latent_dim: int = hyperparams.latent_dim
        self.input_channels: int = hyperparams.input_channels
        self.input_length: int = hyperparams.input_length
        self.hidden_dims: list[int] = list(hyperparams.hidden_dims)
        
        # Build encoder blocks using the provided block type.
        encoder_blocks: list[nn.Module] = []
        in_ch: int = self.input_channels
        for out_ch in self.hidden_dims:
            encoder_blocks.append(
                Resnet1dBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    mode=hyperparams.block_type_encoder,
                    activation=activation
                )
            )
            in_ch = out_ch
        self.encoder: nn.Sequential = nn.Sequential(*encoder_blocks)
        
        # Compute encoded length using:
        # L_out = floor((L_in + 2*padding - kernel_size) / stride) + 1
        L_enc: int = self.input_length
        for _ in self.hidden_dims:
            L_enc = (L_enc + 2 * padding - kernel_size) // stride + 1
        self.L_enc: int = L_enc
        self.flatten_dim: int = self.hidden_dims[-1] * self.L_enc
        
        # Fully connected layers to produce latent variables.
        self.fc_mu: nn.Linear = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar: nn.Linear = nn.Linear(self.flatten_dim, self.latent_dim)
        
        # FC layer to project from latent space back to decoder input.
        self.fc_decoder: nn.Linear = nn.Linear(self.latent_dim, self.flatten_dim)
        
        # Build decoder blocks using the provided block type.
        decoder_blocks: list[nn.Module] = []
        rev_hidden: list[int] = list(reversed(self.hidden_dims))
        in_ch = rev_hidden[0]
        for out_ch in rev_hidden[1:]:
            decoder_blocks.append(
                Resnet1dBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    mode=hyperparams.block_type_decoder,
                    activation=activation
                )
            )
            in_ch = out_ch
        self.decoder: nn.Sequential = nn.Sequential(*decoder_blocks)
        
        # Final layer to map features back to original input channels.
        self.final_layer: nn.Conv1d = nn.Conv1d(
            in_channels=in_ch,
            out_channels=self.input_channels,
            kernel_size=1
        )
        
        # Simulate the decoded output length.
        L_dec: int = self.L_enc
        num_decoder_blocks: int = len(rev_hidden) - 1
        for _ in range(num_decoder_blocks):
            # Each decoder block: L_out = (L_in - 1) * stride - 2*padding + kernel_size
            L_dec = (L_dec - 1) * stride - 2 * padding + kernel_size
        L_final: int = L_dec  # final_layer does not change length.
        
        # Compute difference to adjust to target input length.
        diff: int = self.input_length - L_final
        if diff > 0:
            # Use an extra ConvTranspose1d to add missing samples.
            self.adjust_layer: nn.Module = nn.ConvTranspose1d(
                self.input_channels,
                self.input_channels,
                kernel_size=diff + 1,
                stride=1,
                padding=0
            )
        else:
            self.adjust_layer: nn.Module = nn.Identity()
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes the input tensor into latent space parameters."""
        x_enc: torch.Tensor = self.encoder(x)  # shape: (batch, hidden_dims[-1], L_enc)
        batch_size: int = x.size(0)
        x_flat: torch.Tensor = x_enc.view(batch_size, -1)
        mu: torch.Tensor = self.fc_mu(x_flat)
        logvar: torch.Tensor = self.fc_logvar(x_flat)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick to sample from the latent distribution."""
        std: torch.Tensor = torch.exp(0.5 * logvar)
        eps: torch.Tensor = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent variable z into a reconstructed audio tensor."""
        batch_size: int = z.size(0)
        x_dec_flat: torch.Tensor = self.fc_decoder(z)
        x_dec: torch.Tensor = x_dec_flat.view(batch_size, self.hidden_dims[-1], self.L_enc)
        x_dec = self.decoder(x_dec)
        x_recon: torch.Tensor = self.final_layer(x_dec)
        # Adjust output length to match input_length.
        x_recon = self.adjust_layer(x_recon)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z: torch.Tensor = self.reparameterize(mu, logvar)
        x_recon: torch.Tensor = self.decode(z)
        return x_recon, mu, logvar
    
    def load_checkpoint(self, checkpoint_path: str, device: torch.device = torch.device("cpu")) -> None:
        """Loads model weights from a checkpoint with error handling."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            print("Checkpoint loaded from:", checkpoint_path)
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
    
    def generate(self, hp:RawHyperParams, num_samples: int = 1 ) -> torch.Tensor:
        """Generates sample audio by decoding random latent vectors."""
        self.to(hp.device)

        z: torch.Tensor = torch.randn(num_samples, self.latent_dim).to(hp.device)
        
        with torch.no_grad():
            generated_audio: torch.Tensor = self.decode(z)
        
        # Ensure the output directory exists.
        os.makedirs(hp.output_audio_dir, exist_ok=True)
        
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save each generated sample as a .wav file.
        for i in range(num_samples):
            # Extract the i-th sample; expected shape (channels, time)
            audio_sample: torch.Tensor = generated_audio[i].cpu()
            filename: str = os.path.join(hp.output_audio_dir, f"gen_audio_{timestamp}_{i}.wav")
            torchaudio.save(filename, audio_sample, hp.sampling_rate)
            print(f"Saved generated sample to {filename}")

        return generated_audio


# if __name__ == '__main__':
#     # Instantiate hyperparameters.
#     hyperparams: RawHyperParams = RawHyperParams()
    
#     # For testing, set a smaller batch size.
#     hyperparams.batch_size = 4
    
#     # Create the ResenetVAE model using hyperparams.
#     model: ResenetVAE = ResenetVAE(hyperparams)
#     model.to(hyperparams.device)
    
#     # Create a dummy batch of audio (shape: [batch, channels, input_length]).
#     dummy_audio: torch.Tensor = torch.randn(
#         hyperparams.batch_size,
#         hyperparams.input_channels,
#         hyperparams.input_length
#     ).to(hyperparams.device)
    
#     # Forward pass.
#     reconstructed, mu, logvar = model(dummy_audio)
#     print("Input shape:", dummy_audio.shape)
#     print("Reconstructed shape:", reconstructed.shape)
#     print("Latent mean shape:", mu.shape)
#     print("Latent logvar shape:", logvar.shape)
    
#     # Generate sample audio.
#     samples: torch.Tensor = model.generate(num_samples=2, device=hyperparams.device)
#     print("Generated sample audio shape:", samples.shape)
