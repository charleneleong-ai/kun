import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import os

# Create output folder
OUTPUT_PATH = './ae_output_v0'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


#### Transforming datasets

# def min_max_normalization(tensor, min_value, max_value):
#     min_tensor = tensor.min()
#     tensor = (tensor - min_tensor)
#     max_tensor = tensor.max()
#     tensor = tensor / max_tensor
#     tensor = tensor * (max_value - min_value) + min_value
#     return tensor

# def tensor_round(tensor):
#     return torch.round(tensor)
# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
#     transforms.Lambda(lambda tensor:tensor_round(tensor))
# ])

# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1].
# Normalisation: norm_img = (image - mean) / std
# eg. minimum value 0 will be converted to (0-0.5)/0.5=-1
# eg. maximum value of 1 will be converted to (1-0.5)/0.5=1.
# The first tuple (0.5, 0.5, 0.5) is the mean for all three channels 
# and the second (0.5, 0.5, 0.5) is the standard deviation for all three channels.

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# MNIST dataset
# Download from torchvision.datasets.mnist
# https://pytorch.org/docs/stable/torchvision/datasets.html#mnist
dataset = MNIST('../',                         # Download dir
        train=True,                         # Download training data only
        transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray (H x W x C) [0, 255]
                                            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        download=True
        )

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())       # classification 

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))

model = autoencoder().to(device)
# criterion = nn.BCELoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

for epoch in range(NUM_EPOCHS):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).to(device)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        MSE_loss = nn.MSELoss()(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
          .format(epoch + 1, NUM_EPOCHS, loss.data, MSE_loss.data))
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        save_image(x, OUTPUT_PATH+'/x_{}.png'.format(epoch))
        save_image(x_hat, OUTPUT_PATH+'/x_hat_{}.png'.format(epoch))

torch.save(model.state_dict(), '/ae_weights.pth')