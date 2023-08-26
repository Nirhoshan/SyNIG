import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt
#%matplotlib inline


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device('cpu')


def to_device(data, device): # move tensors to the chosen device
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader(): #Wrap a dataloader to move data to a device

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self): # Yield a batch of data after moving it to device
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self): # Number of batches
        return len(self.dl)



import torch.nn as nn

discriminator = nn.Sequential(

    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=2, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
)



latent_size = 64

generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2, bias=False),
    nn.Tanh()
)

def weight_init(m): # weight_initialization
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0,0.02)




def train_discriminator(real_images, opt_d):
    opt_d.zero_grad()

    real_images=real_images.float()
    real_preds = discriminator(real_images)
    real_loss = -torch.mean(real_preds)

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_preds = discriminator(fake_images)
    fake_loss = torch.mean(fake_preds)

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_loss.item(), fake_loss.item()


def train_generator(opt_g):
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    loss = -torch.mean(preds)
    loss.backward()
    opt_g.step()

    return loss.item()


eud=[]
cos_similarity=[]
ssim_scores=[]
mses=[]
mfids=[]

def save_samples(index, epochs):
    fake_no=400 # Number of samples to be generated
    for j in range(fake_no):
        latent_tensors = torch.randn(1, latent_size, 1, 1, device=device)
        fake_images = generator(latent_tensors)
        if index==epochs:
            fake_fname = f'generated-csv-{str(index).zfill(5)}-{str(j).zfill(5)}.csv'
            gasf_csv = pd.DataFrame(data=fake_images[0][0].cpu().detach().numpy())
            gasf_csv.to_csv(os.path.join(sample_dir, fake_fname), header=False, index=False)
            print('Saving', fake_fname)


def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
    opt_g = torch.optim.RMSprop(generator.parameters(), lr=lr)

    for epoch in range(epochs):
        for real_images in tqdm(train_dl):
            # Train discriminator
            for parm in discriminator.parameters():
                parm.data.clamp_(-clamp_num, clamp_num)
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

        # Save generated images
        save_samples(epoch + start_idx, epochs)

    return losses_g, losses_d, real_scores, fake_scores


if __name__ == '__main__':
    DATA_DIR = 'Youtube/vid1' # directory where the input csvs are stored
    sample_dir = 'Youtube_synth/vid1' # directory to save the generated csvs
    epochs = 1500
    traces = 80
    print(DATA_DIR)
    image_size = 125
    batch_size =8

    train_ds=[]
    i = 1
    for file in os.scandir(DATA_DIR):
        df = pd.read_csv(file, header=None).values
        data = torch.from_numpy(np.array([df]))
        train_ds.append(data)
        i = i + 1
        if i == (traces + 1):
            break
    print(len(train_ds))
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)

    discriminator = to_device(discriminator, device)

    discriminator.apply(weight_init)
    generator.apply(weight_init)

    xb = torch.randn(batch_size, latent_size, 1, 1)  # random latent tensors
    fake_images = generator(xb)
    print(fake_images.shape)

    generator = to_device(generator, device)

    os.makedirs(sample_dir, exist_ok=True)

    fixed_latent = torch.randn(1, latent_size, 1, 1, device=device)

    lr = 0.00005 # learning rate
    clamp_num=0.01 # WGAN clip gradient
    history = fit(epochs, lr)

    losses_g, losses_d, real_scores, fake_scores = history

    import os

    files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]
    files.sort()

    plt.clf()
    plt.plot(losses_d, '-')
    plt.plot(losses_g, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.savefig("Loss_1.png")
