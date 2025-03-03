import torch 
import torch.nn as nn 
from typing import Tuple, Optional
from enocder_decoder_blocks import Encoder, Decoder
from hyperparams import TVAEParams
from utils import get_sinusoidal_positional_encoding

class TVAE(nn.Module):
    def __init__(self, hp:TVAEParams):
        super().__init__()
        self.checkpoint_loaded = False

        self.encoder = Encoder(hp=hp)
        self.decoder = Decoder(hp=hp)

    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor)->torch.Tensor:
        # z = mu + sigma * eps
        eps = torch.randn_like(mu)
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * eps

    def forward(self, x:torch.Tensor, tgt: torch.Tensor)->tuple:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z, tgt)
        return reconstruction, mu, logvar
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.encoder.pos_embedding.device)
            self.load_state_dict(checkpoint)
            self.checkpoint_loaded = True
            print(f"Checkpoint loaded successfully from: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")

    
    def generate_music(self, hp: TVAEParams, batch_size: int = 1, max_len: Optional[int] = None) -> torch.Tensor:
        """
        Returns:
            Generated audio tensor of shape [batch_size, max_len, 1].
        """
        device = next(self.parameters()).device
        if max_len is None:
            max_len = hp.input_size[0]

        try:
            # Sample a latent vector z ~ N(0, I)
            z = torch.randn(batch_size, hp.latent_dim, device=device)
        except Exception as e:
            print(f"Error generating latent vector: {e}")
            raise e

        # Prepare memory for cross-attention.
        latent = self.decoder.latent_transformation(z)  # [B, d_model]
        memory = latent.unsqueeze(0)  # [1, B, d_model]

        # Precompute sinusoidal positional encodings for max_len positions.
        pos_enc_all = get_sinusoidal_positional_encoding(max_len, hp.d_model, device=device)  # [max_len, d_model]

        generated_tokens = []

        # Autoregressive generation loop:
        # At each time step, the decoder receives all tokens generated so far.
        for t in range(max_len):
            if t == 0:
                # For t==0, create an initial token (shape: [1, B, d_model])
                current_token = torch.zeros(1, batch_size, hp.d_model, device=device)
                # Add sinusoidal positional encoding for position 0.
                current_token = current_token + pos_enc_all[0].unsqueeze(0)
                current_target = current_token
            else:
                # Concatenate all previously generated tokens.
                current_target = torch.cat(generated_tokens, dim=0)  # [t, B, d_model]
                # Create a new token placeholder for the current time step.
                new_token = torch.zeros(1, batch_size, hp.d_model, device=device)
                # Add sinusoidal positional encoding for position t.
                new_token = new_token + pos_enc_all[t].unsqueeze(0)
                # Append the new token to the current sequence.
                current_target = torch.cat([current_target, new_token], dim=0)  # [t+1, B, d_model]

            # Create a causal mask for the current target sequence length.
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_target.size(0)).to(device)
            # Pass the current target sequence through the decoder.
            decoder_out = self.decoder.transformer_decoder(tgt=current_target, memory=memory, tgt_mask=tgt_mask)  # [seq_len, B, d_model]
            # For autoregressive generation, take the output corresponding to the last token.
            current_output = decoder_out[-1:, :, :]  # Shape: [1, B, d_model]
            generated_tokens.append(current_output)

        # Concatenate all generated tokens: shape [max_len, B, d_model]
        generated_sequence = torch.cat(generated_tokens, dim=0)
        # Transpose to [B, max_len, d_model]
        generated_sequence = generated_sequence.permute(1, 0, 2)
        # Project the generated sequence to output space: [B, max_len, 1]
        output = self.decoder.out_projection(generated_sequence)
        return output



# if __name__ == '__main__':

#     # Create hyperparameters instance
#     hp = TVAEParams()
    
#     # Instantiate the model and move it to the appropriate device
#     model = TVAE(hp)
#     device = hp.device
#     model.to(device)
    
#     # Create a dummy input:
#     #[batch_size, channels, sequence_length]
#     channels = hp.input_channels 
#     seq_len = hp.input_size[0]    
#     dummy_input = torch.randn(hp.batch_size, channels, seq_len).to(device)
#     print("Dummy Input shape:", dummy_input.shape)

#     total_params = sum(p.numel() for p in model.parameters())
#     print("Total model parameters:", total_params)
#     print(model)
#     # Forward pass through the model
#     reconstruction, mu, logvar = model(dummy_input, dummy_input)
    
#     # Print out the shapes of the outputs to verify the dimensions
#                # Expected: [B, channels, seq_len]
#     print("Reconstruction shape:", reconstruction.shape)     # Expected: [B, seq_len, 1]
#     print("Mu shape:", mu.shape)                             # Expected: [B, latent_dim]
#     print("Logvar shape:", logvar.shape)        
        