import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd

class YogaPosesDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None) -> None:
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def _create_metadata(data_path: Path) -> None:
        data = []
        if not data_path.exists():
            raise FileNotFoundError(f"No folder named {data_path}")
        else:
            for folder in data_path.iterdir():
                if folder.is_dir():
                    for file in folder.iterdir():
                        data.append({"filepath":file,"class":folder.name})
        return pd.DataFrame(data)
