import os
from hyperparams import HyperParams
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SpectrogramDataset(Dataset):
    def __init__(self, hyper_params: HyperParams, transform:transforms = None):
        super().__init__()
        """
        Args:
            hyper_params.spectrogram_data_dir (str): Root directory containing spectrogram images in subfolders.
            hyper_params.image_size (tuple): Desired image size (H, W) for resizing.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_apth = hyper_params.spectrogram_data_dir
        self.image_size = hyper_params.image_size
        self.image_files = []
        self.labels = []

        for genre in os.listdir(hyper_params.spectrogram_data_dir):
            genre_dir = os.path.join(hyper_params.spectrogram_data_dir, genre)
            if os.path.isdir(genre_dir):
                for f in os.listdir(genre_dir):
                    if f.lower().endswith('.png'):
                        self.image_files.append(os.path.join(genre_dir, f))
                        # label is optional
                        self.labels.append(genre)

        # create if no taransforms is provided 
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self)->int:
        return len(self.image_files)
    
    def __getitem__(self, index):
        # If index is a slice, return a list of samples.
        if isinstance(index, slice):
            indices = list(range(*index.indices(len(self))))
            return [self[i] for i in indices]
        
        # If index is a list or tuple, handle similarly:
        if isinstance(index, (list, tuple)):
            return [self[i] for i in index]
        
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        label = self.labels[index]
        return image, label

