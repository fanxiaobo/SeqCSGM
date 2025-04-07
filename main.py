import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np

from diffusers.models import AutoencoderKLTemporalDecoder
from seqcsgm import autoencoder_estimator
from mom import HParams as MomParams, mom_estimator
from e2e import HParams as e2eParams, E2EAutoencoder
from lasso import lasso_estimator_mmnist, lasso_wavelet_estimator
from UCF_handler import UCF101VideoDataset
from MovingMNISTVideoLoader import MovingMNIST
from utils import *

# load test data
def test_data_load(dataset, batch_size):
    if dataset=="ucf":
        dataset = UCF101VideoDataset(video_folder="../data/UCF101", transform=None, start=0, end=10)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif dataset=="mmnist":
        train_set = MovingMNIST(root='../data/Movingmnist', start=0, end=10, train=True, download=True)
        test_set = MovingMNIST(root='../data/Movingmnist', start=0, end=10, train=False, download=True)  
        train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=True)
    else:
        test_loader = None
    return test_loader



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ucf", choices=["ucf","mmnist"], help="Dataset to use (ucf or mmnist)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for test")
    parser.add_argument("--llambda", type=float, default=0.1, help="Lambda value for the loss function")
    args = parser.parse_args()
    
    dataset = args.dataset
    batch_size = args.batch_size
    llambda = args.llambda

    
    img_size = 64  
    latent_dim = 4 * 8 * 8  
    learning_rate = 0.05  
    gif_dir = None  
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    noise_std = 0.1
    channels=1 if dataset=="mmnist" else 3
    T = 8                   # Total number of time slots
    
    # pre-trained model
    vae = AutoencoderKLTemporalDecoder.from_pretrained("../models", subfolder="vae_temporal_decoder", torch_dtype=torch.float32)
    vae.to(device)
    vae.eval()
    
    for param in vae.parameters():               
        param.requires_grad = False
 
    num_measurements = [1500 if i == 0  else 300 for i in range(T)]  # Set a larger initial moment to obtain accurate estimation                                                                  
    max_update_iter = [2500 if i == 0 else 300 for i in range(T)]   
   

    # Obtain a batch of real images
    test_loader = test_data_load(dataset, batch_size)
    if dataset=="mmnist":
        real_images_T, _ = next(iter(test_loader))                  #  [batch_size, t, channels, height, width]
    else:
        real_images_T = next(iter(test_loader))                  #  [batch_size, t, channels, height, width]
    real_images_T = real_images_T.to(device)
    
    # t = 0
    A0 = torch.randn(num_measurements[0], img_size * img_size * channels, device=device)
    # t>0
    A1 = torch.randn(num_measurements[1], img_size * img_size * channels, device=device)
    z0_batch = None
    lasso_weights =  [np.ones(channels*img_size*img_size)]*batch_size
    
    error_records = {
        "seqcsgm": {t: [] for t in range(T)},
        "csgm": {t: [] for t in range(T)},
        "mom": {t: [] for t in range(T)},
        "e2e": {t: [] for t in range(T)},
        "modifiedCS": {t: [] for t in range(T)}
    }
    # Initialize the saved image array
    estimated_images_seqcsgm = torch.zeros(batch_size, T, channels, img_size, img_size, device=device)
    estimated_images_csgm = torch.zeros(batch_size, T, channels, img_size, img_size, device=device)
    estimated_images_mom = torch.zeros(batch_size, T, channels, img_size, img_size, device=device)
    estimated_images_e2e = torch.zeros(batch_size, T, channels, img_size, img_size, device=device)
    estimated_images_modifiedCS = torch.zeros(batch_size, T, channels, img_size, img_size, device=device)
    for t in range(T):
        # measurement y
        A = A0 if t==0 else A1
        real_images = real_images_T[:, t, :, :, :]          
        y_batch_val = torch.matmul(real_images.view(batch_size, -1), A.t()) 
        y_batch_val = y_batch_val + torch.normal(mean=0, std=noise_std, size=y_batch_val.shape, device=device)       # add measurement noise

        # seqcsgm
        print(f"Time step: {t}, Starting seqcsgm estimation...")
        estimator = autoencoder_estimator(vae, img_size, latent_dim, channels, A, batch_size,
                                          learning_rate, max_update_iter[t], gif_dir, dataset, device)
        estimated_images, z0_batch = estimator(y_batch_val, z0_batch=z0_batch, z0_weight=llambda)
        estimated_images_seqcsgm[:, t, :, :, :] = estimated_images.view(batch_size,-1,img_size,img_size)
 
        # csgm
        print(f"Time step: {t}, Starting csgm estimation...")
        estimator_csgm = autoencoder_estimator(vae, img_size, latent_dim, channels, A, batch_size,
                                          learning_rate, max_update_iter[0], gif_dir, dataset, device)
        estimated_images, _ = estimator_csgm(y_batch_val)
        estimated_images_csgm[:, t, :, :, :] = estimated_images.view(batch_size,-1,img_size,img_size)
        
        # mom
        print(f"Time step: {t}, Starting mom estimation...")
        hparams_mom = MomParams(num_measurements[t], batch_size, learning_rate, max_update_iter[0], 
                                dataset, img_size, noise_std, mom_batch_size=20, device=device)
        estimator_mom = mom_estimator(vae, hparams_mom)
        estimated_images, _ = estimator_mom(A.t(), y_batch_val, hparams_mom)
        estimated_images_mom[:, t, :, :, :] = estimated_images.view(batch_size,-1,img_size,img_size)
        
        
        """
        # The input data for e2e and modifiedCS is [0 1]
        """
        real_images = (real_images + 1) / 2 
        y_batch_val = torch.matmul(real_images.view(batch_size, -1), A.t()) 
        y_batch_val = y_batch_val + torch.normal(mean=0, std=noise_std, size=y_batch_val.shape, device=device)
        
        # E2E
        print(f"Time step: {t}, Starting e2e estimation...")
        hparams_e2e = e2eParams(num_measurements[t], batch_size, dataset, img_size)
        estimator_e2e = E2EAutoencoder(hparams_e2e).to(device)
        if dataset=="mmnist":
            if t==0:
                estimator_e2e.load_state_dict(torch.load("../models/e2e_movingmnist_m1500model.pth", map_location=device))
            else:
                estimator_e2e.load_state_dict(torch.load("../models/e2e_movingmnist_m300model.pth", map_location=device))
        else:
            if t==0:
                estimator_e2e.load_state_dict(torch.load("../models/e2e_ucf_m2000model.pth", map_location=device))
            else:
                estimator_e2e.load_state_dict(torch.load("../models/e2e_ucf_m500model.pth", map_location=device))
        estimated_images_e2e[:, t, :, :, :] = estimator_e2e(real_images)
        
        # modifiedCS
        print(f"Time step: {t}, Starting modifiedCS estimation...")
        if dataset=="mmnist":
            modifiedCS = lasso_estimator_mmnist(llambda)
        else:
            modifiedCS = lasso_wavelet_estimator(llambda)
        estimated_images, lasso_weights = modifiedCS(A.cpu().numpy(), y_batch_val=y_batch_val.cpu().numpy(), lasso_weights=lasso_weights)
        estimated_images_modifiedCS[:, t, :, :, :] = torch.tensor(estimated_images, device=device).view(batch_size, -1, img_size, img_size)

        
        torch.cuda.empty_cache()  
        
        
    save_images(real_images=real_images_T, save_dir="save_dir", seqcsgm=estimated_images_seqcsgm, csgm=estimated_images_csgm, 
                mom=estimated_images_mom, e2e=estimated_images_e2e, lasso=estimated_images_modifiedCS)