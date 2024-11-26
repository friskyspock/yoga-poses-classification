import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image

class YogaPosesDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, target_transform=None) -> None:
        self.img_labels = self._create_metadata(Path(data_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def _create_metadata(self, data_path: Path) -> None:
        data = []
        if not data_path.exists():
            raise FileNotFoundError(f"No folder named {data_path}")
        else:
            for folder in data_path.iterdir():
                if folder.is_dir():
                    for file in folder.iterdir():
                        data.append({"filepath":file,"class":folder.name})
        return pd.DataFrame(data)

if __name__ == "__main__":
    data = YogaPosesDataset("data",None,None)
    print(data[0])