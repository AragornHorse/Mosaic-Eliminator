import fiona
import torch
import torch.nn as nn
import torch.optim as optim

# 64 -> 256

class Eliminator(nn.Module):
    def __init__(self):
        super(Eliminator, self).__init__()

        self.attn1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=4, padding=0)
        )

        self.attn2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=4, padding=0)
        )

        self.attn3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2, padding=1)
        )

        self.attn4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=7, stride=4, padding=3),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=7, stride=4, padding=3),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=8, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=8, stride=4, padding=2),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2, padding=1),
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        attn1 = self.attn1(x)
        attn2 = self.attn2(x)
        attn3 = self.attn3(x)
        attn4 = self.attn4(x)
        out = self.relu(attn1 + attn2 + attn3 + attn4)

        return out

        # return torch.sigmoid(attn1 + attn2 + attn3 + attn4) * 255


class AI:
    def __init__(self, device=torch.device("cpu"), lr=1e-2):
        self.device = device
        self.eliminator = Eliminator().to(device)

        self.loss_func = nn.MSELoss()
        self.opt = optim.Adam(self.eliminator.parameters(), lr=lr)

        self.schedular = optim.lr_scheduler.StepLR(self.opt, gamma=0.9, step_size=1)

    def train(self, data):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.eliminator(x)

        loss = self.loss_func(out, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss

    def eval(self, data):
        x, y = data
        x = x.to(self.device)

        out = self.eliminator(x)

        return out

    def eval_and_show(self, x):
        x = x.unsqueeze(0).to(self.device)
        out = self.eliminator(x)

        import dataset
        dataset.show_data(out)







