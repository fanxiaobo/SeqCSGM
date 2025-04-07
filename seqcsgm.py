import torch
import torch.optim as optim
import os
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

def autoencoder_estimator(gen_model, img_size, latent_dim, channels, A,
 batch_size, learning_rate, max_update_iter, gif_dir=None, dataset="mmnist", device="cuda"):
    vae = gen_model

    def loss_function(y_batch, y_hat_batch, z_batch, mloss1_weight, mloss2_weight, zprior_weight, z0_batch, z0_weight):
        m_loss1 = torch.mean(torch.abs(y_batch - y_hat_batch), dim=1)
        m_loss2 = torch.mean((y_batch - y_hat_batch) ** 2, dim=1)
        zp_loss = torch.sum(z_batch ** 2, dim=[1, 2, 3])   
        total_loss = mloss1_weight * m_loss1 + mloss2_weight * m_loss2 + zprior_weight * zp_loss
        if z0_batch is not None:                                     # recovery estimation: If z has a reference value, add it to the loss function ||z-z0||^2
            total_loss += z0_weight*torch.sum((z_batch-z0_batch) ** 2, dim=[1, 2, 3])
        return total_loss.mean()

    def estimator(y_batch_val, mloss1_weight=1.0, mloss2_weight=1.0, zprior_weight=0.001*0, z0_batch=None, z0_weight=100):
        best_keeper = {"best_loss": float("inf"), "best_z": None}
        
        # Define optimization variables
        if z0_batch is None:
            z_batch = torch.randn(
            batch_size, 4, 8, 8, 
            dtype=torch.float32,    
            requires_grad=True, 
            device=device
            )
        else:
            z_batch = z0_batch.clone().detach().requires_grad_(True)                    # If z has a reference value, initialize it using it
        optimizer = optim.Adam([z_batch], lr=learning_rate)
            
        for i in range(max_update_iter):
            optimizer.zero_grad()
            
            # Generate image
            x_hat_batch = vae.decode(z_batch, 1).sample 
            if dataset=="mmnist":
                x_hat_batch = 0.2989 * x_hat_batch[:, 0, :, :] + 0.5870 * x_hat_batch[:, 1, :, :] + 0.1140 * x_hat_batch[:, 2, :, :]
            
            # measurements
            y_hat_batch = torch.matmul(x_hat_batch.view(batch_size, -1), A.t())

            # loss
            total_loss = loss_function(y_batch_val, y_hat_batch, z_batch, mloss1_weight, mloss2_weight, zprior_weight, z0_batch, z0_weight)

        
            # backward
            total_loss.backward()

            optimizer.step()

#             print(f"Iteration {i + 1}/{max_update_iter}, Loss: {total_loss.item():.4f}")

            # Save Best Results
            if total_loss.item() < best_keeper["best_loss"]:
                best_keeper["best_loss"] = total_loss.item()
                best_keeper["best_z"] = z_batch.clone().detach()

            # save GIF
            if gif_dir and (i % 100 == 0):
                images = x_hat_batch.detach()
                save_dir = os.path.join(gif_dir, "iter_{}".format(i))
                os.makedirs(save_dir, exist_ok=True)
                save_image(images, os.path.join(save_dir, "images.png"), nrow=5, normalize=True)

        # return Best Results
        best_z = best_keeper["best_z"]
        best_x_hat = vae.decode(best_z, 1).sample  # Generate the best image using the decoder
        if dataset=="mmnist":
            best_x_hat = 0.2989 * best_x_hat[:, 0, :, :] + 0.5870 * best_x_hat[:, 1, :, :] + 0.1140 * best_x_hat[:, 2, :, :]
        return best_x_hat, best_z

    return estimator