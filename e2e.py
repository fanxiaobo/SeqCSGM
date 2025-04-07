import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader



class HParams:
    def __init__(self, num_measurements=1500, test_batch_size=10, dataset="mmnist", img_size=64):
        self.num_measurements = num_measurements
        self.dataset = dataset
        self.img_size = img_size
        self.max_train_steps = 50000
        self.summary_iter = 1000
        self.layer_sizes = [50, 200]
        self.train_batch_size = 8
        self.test_batch_size = test_batch_size
        self.learning_rate = 0.001
        self.is_A_trainable = False
        self.noise_std = 0.1


# model define for moving mnist
class E2EAutoencoderMmnist(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.input_dim = 64 * 64  # Moving MNIST and UCF 101: 64x64
        self.A = nn.Linear(self.input_dim, hparams.num_measurements, bias=False)
        self.A.weight.data.normal_(std=1.0/hparams.num_measurements)
        self.A.weight.requires_grad = hparams.is_A_trainable
        
        layers = []
        prev_size = hparams.num_measurements
        for size in hparams.layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        self.encoder = nn.Sequential(*layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(prev_size, self.input_dim),
            nn.Sigmoid()
        )
        # Noise
        self.noise_std = hparams.noise_std

    def forward(self, x):
        x_flat = x.view(x.size(0),1, -1)  # [B*T, 64*64]
        y = self.A(x_flat)               
        
        if not self.training:  # Only add noise during testing
            noise = torch.normal(mean=0, std=self.noise_std, size=y.size(), device=y.device)
            y += noise
        hidden = self.encoder(y)
        recon = self.decoder(hidden)     
        return recon.view_as(x)          
    
    
 # model define for UCF
class E2EAutoencoderUCF(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # Modify the input dimension to the size of a three channel image
        self.input_dim = 3 * 64 * 64  # channel*64*64
        self.A = nn.Linear(self.input_dim, hparams.num_measurements, bias=False)
        # A is initialized with a mean of 0 and a variance of 1
        self.A.weight.data = torch.randn_like(self.A.weight.data)
        self.A.weight.requires_grad = hparams.is_A_trainable
        
        layers = []
        prev_size = hparams.num_measurements
        for size in hparams.layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        self.encoder = nn.Sequential(*layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(prev_size, self.input_dim),
            nn.Sigmoid()
        )
        # noise
        self.noise_std = hparams.noise_std

    def forward(self, x):   
        x_flat = x.view(x.size(0), -1)  # [B, 3*64*64]
        y = self.A(x_flat)              
        if not self.training:  # Only add noise during testing
            noise = torch.normal(mean=0, std=self.noise_std, size=y.size(), device=y.device)
            y += noise
        hidden = self.encoder(y)
        recon = self.decoder(hidden)     
        return recon.view_as(x)          
    
def E2EAutoencoder(hparams):
    if hparams.dataset=="mmnist":
        return E2EAutoencoderMmnist(hparams)
    elif hparams.dataset=="ucf":
        return E2EAutoencoderUCF(hparams)
    else:
        return None