import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasetweb import SegmentationDataset

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_url, batch_size=32, num_workers=0):
        super().__init__()
        self.data_url = data_url
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = T.Compose([
            T.Lambda(lambda x: x.float() / 10000.0),
        ])

        self.train_dataset = SegmentationDataset(self.data_url, 
                                                 split='train', 
                                                 transform=transform)
        self.val_dataset = SegmentationDataset(self.data_url, 
                                               split='validation', 
                                               transform=transform)
        self.test_dataset = SegmentationDataset(self.data_url, 
                                                split='test', 
                                                transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)


if __name__ == "__main__":
    path = "https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus/resolve/main"
    datamodule = SegmentationDataModule(path, batch_size=4)
    datamodule.setup()
    train_data_loader = datamodule.train_dataloader()
    image_crops, target_crops = next(iter(train_data_loader))
    print(image_crops.shape, target_crops.shape)
