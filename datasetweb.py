import torch
from torch.utils.data import Dataset
import mlstac
from einops import rearrange

class SegmentationDataset(Dataset):
    def __init__(self, data_url, split = 'train', transform=None, crop_size=1024):
        self.data_url = data_url
        self.split = split
        self.transform = transform
        self.crop_size = crop_size

        self.data_url = f"{self.data_url}/{self.split}/{self.split}_2000_high.mlstac"

    def __len__(self):
        self.metadata = mlstac.load_metadata(self.data_url)
        return len(self.metadata)

    def __getitem__(self, idx):
        data = mlstac.load_data(self.metadata[idx:idx+1], quiet=True)
        data = data.squeeze(0)

        image = data[:-2, :, :].astype('float32')
        target = data[-1, :, :].astype('int64')

        image = torch.tensor(image)
        target = torch.tensor(target)

        image = image.permute(1, 2, 0)

        image_crops = rearrange(image, 
                                '(h ch) (w cw) c -> (h w) c ch cw', 
                                ch=self.crop_size, 
                                cw=self.crop_size)
        target_crops = rearrange(target, 
                                 '(h ch) (w cw) -> (h w) ch cw', 
                                 ch=self.crop_size, 
                                 cw=self.crop_size)

        rand_idx = torch.randint(0, image_crops.size(0), (1,)).item()
        image_crop = image_crops[rand_idx]
        target_crop = target_crops[rand_idx]

        if self.transform:
            image_crop = self.transform(image_crop)

        return image_crop, target_crop
