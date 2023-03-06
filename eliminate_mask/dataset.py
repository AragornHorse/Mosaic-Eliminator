import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
# import random
# # import torchvision.transforms as transforms


def show_data(img):
    import matplotlib.pyplot as plt
    if img.dim() > 3:
        img = img[0]
    img = img.cpu().detach().numpy().transpose([1, 2, 0]).astype(int)
    plt.imshow(img)
    plt.show()


class Data(Dataset):
    def __init__(self, path=None):
        if path is None:
            path = r"C:\Users\DELL\Desktop\datasets\mask"

        lst = glob.glob(path + r"\mask\*.npy")
        self.lst = [name.split("\\")[-1] for name in lst]
        self.mask_path = path + r"\mask\\"
        self.no_mask_path = path + r"\no_mask\\"

    def __getitem__(self, item):
        if item == 0:
            item = random.randint(1, 10)
        name = self.lst[item]
        x = np.load(self.mask_path + name)
        x = torch.tensor(x, dtype=torch.float)

        y = np.load(self.no_mask_path + name)
        y = torch.tensor(y, dtype=torch.float)

        return x, y

    def __len__(self):
        return len(self.lst)

# d = Data()
# for x, y in d:
#     print(x.size(), y.size())
#     show_data(x)
#     show_data(y)

def load(batch_size=4):
    loader = DataLoader(Data(), batch_size=batch_size, shuffle=True)
    return loader












