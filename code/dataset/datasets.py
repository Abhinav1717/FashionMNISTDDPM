from torch.utils.data import Dataset
import os
from PIL import Image

class FashionMNISTDataset(Dataset):
    
    def __init__(self,images_directory_path,transform=None):
        
        self.images_directory_path = images_directory_path
        self.transform = transform
        
        self.filenames = os.listdir(images_directory_path)
    def __len__(self):
        
        return len(self.filenames)
    
    def __getitem__(self, index):
        
        file_name = self.filenames[index]
        image_path = os.path.join(self.images_directory_path,file_name)
        image = Image.open(image_path)
        sample = image
        if self.transform:
            sample = self.transform(sample)
        
        return sample