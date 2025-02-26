import torch
import torch.nn as nn

class Resnet1dBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, 
                 kernel_size:int = 3, stride:int = 1, padding:int = 1, 
                 mode:str = 'encoder', activation:nn.Module = nn.ReLU(inplace=True)):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride for the convolution.
            padding (int): Padding for the convolution.
            mode (str): 'encoder' for Conv1d or 'decoder' for ConvTranspose1d.
            activation (nn.Module): Activation function.
        
        The block applies two convolutional (or transposed convolutional) layers sequentially
        and adds a skip connection from the input to the output.
        """
        super().__init__()
        self.mode = mode
        self.activation = activation

        if mode == 'encoder':
            self.block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm1d(out_channels),
                activation,
                nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                nn.BatchNorm1d(out_channels)
            )
        elif mode == 'decoder':
            self.block = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm1d(out_channels),
                activation,
                nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                nn.BatchNorm1d(out_channels)
            )
        else:
            raise ValueError("Mode must be either 'encoder' or 'decoder'")

        # Create a downsampling layer for the skip connection if dimensions don't match.
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            if mode == 'encoder':
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels)
                )
            elif mode == 'decoder':
                self.downsample = nn.Sequential(
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels)
                )

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        # Combine the skip connection and apply activation.
        out += identity
        out = self.activation(out)
        return out


# if __name__ == '__main__':
#     # Encoder block: maps (batch, 16, L) -> (batch, 32, new_length)
#     encoder_block = Resnet1dBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, mode='encoder')
#     x = torch.randn(8, 16, 100) 
#     encoder_out = encoder_block(x)
#     print("Encoder output shape:", encoder_out.shape)

#     # Decoder block: maps (batch, 32, L) -> (batch, 16, new_length)
#     decoder_block = Resnet1dBlock(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, mode='decoder')
#     x = torch.randn(8, 32, 50)  
#     decoder_out = decoder_block(x)
#     print("Decoder output shape:", decoder_out.shape)
