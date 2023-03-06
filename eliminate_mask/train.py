import matplotlib.pyplot as plt
import torch
from model import AI
from dataset import load, Data, show_data

loader = load(batch_size=8)

model = AI(device=torch.device("cuda"), lr=1e-3)

for _ in range(10):
    for epoch in range(1000):
        for data in loader:
            loss = model.train(data)
            print(loss)

    data = Data()

    show_data(data[0][0])
    show_data(data[0][1])
    model.eval_and_show(data[0][0])












