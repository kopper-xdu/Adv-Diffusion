from torch.utils.data import Dataset
import os
from PIL import Image


class base_dataset(Dataset):
    def __init__(self, dir, transform=None) -> None:
        super().__init__()
        
        self.dir = dir
        self.img_names = sorted(os.listdir(os.path.join(dir, 'src')))
        self.tgt_img_names = sorted(os.listdir(os.path.join(dir, 'target')))
        # print(self.img_names)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.dir, 'src', self.img_names[index])
        img = Image.open(img_path).convert('RGB')
        
        tgt_index = index % len(self.tgt_img_names)
        tgt_img_path = os.path.join(self.dir, 'target', self.tgt_img_names[tgt_index])
        tgt_img = Image.open(tgt_img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            tgt_img = self.transform(tgt_img)
            
        return img, tgt_img