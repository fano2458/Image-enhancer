import torch
import glob as glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from srcnn import SRCNN
from PIL import Image
from utils import psnr
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_data = data[0].to(device)
            label = data[1].to(device)
            outputs = model(image_data)
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr

    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr


class SRCNNDataset(Dataset):
    def __init__(self, image_paths):
        self.all_image_paths = glob.glob(f"{image_paths}/*")

    def __len__(self):
        return (len(self.all_image_paths))

    def __getitem__(self, index):
        label = Image.open(self.all_image_paths[index]).convert('RGB')
        w, h = label.size[:]
        low_res_img = label.resize((int(w*0.5), int(h*0.5)), Image.BICUBIC)
        image = low_res_img.resize((w, h), Image.BICUBIC)

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        image /= 255.
        label /= 255.

        image = image.transpose([2, 0, 1])
        label = label.transpose([2, 0, 1])

        return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.float)
        

def get_datasets(image_paths):
    dataset_test = SRCNNDataset(image_paths)
    return dataset_test


def get_dataloaders(dataset_test):
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
    return test_loader


if __name__ == '__main__':
    model = SRCNN().to(device)
    model.load_state_dict(torch.load('../outputs/model.pth'))

    data_paths = [['../input/Set5/original', 'Set5'], ['../input/Set14/original', 'Set14']]

    for data_path in data_paths:
        dataset_test = get_datasets(data_path[0])
        test_loader = get_dataloaders(dataset_test)

        _, test_psnr = validate(model, test_loader, device)
        print(f"Test PSNR on {data_path[1]}: {test_psnr:.3f}")
