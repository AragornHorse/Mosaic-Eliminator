import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

dir_path = r"C:\Users\DELL\Desktop\datasets\mask"

mask_path = dir_path + r"\mask"
no_mask_path = dir_path + r"\no_mask"
img_path = dir_path + r"\img"

img_list = glob.glob(img_path + r"\*")
num = 0

for path in img_list:
    img = Image.open(path).resize((256, 256))

    np.save(no_mask_path + r"\{}.npy".format(num), np.array(img).transpose([2, 0, 1]))
    print(np.array(img).transpose([2, 0, 1]).shape)

    img.thumbnail((64, 64), Image.ANTIALIAS)
    np.save(mask_path + r"\{}.npy".format(num), np.array(img).transpose([2, 0, 1]))
    print(np.array(img).transpose([2, 0, 1]).shape)
    num += 1




