# Using MOM minimization: “Robust Compressed Sensing using Generative Models”. 
# Directly using alg1 mentioned in the article yields poor results, possibly due to Gaussian reasons
import numpy as np
import torch
import torch.optim as optim

class HParams:
    def __init__(self, num_measurements=1500, batch_size=8, learning_rate=0.1,
    max_update_iter=2000, dataset="mmnist", img_size=64, noise_std=0.1, mom_batch_size=20, device='cuda'):
        self.num_measurements = num_measurements
        self.batch_size = batch_size
        self.mom_batch_size = mom_batch_size
        self.learning_rate = learning_rate
        self.max_update_iter = max_update_iter
        self.img_size = img_size
        self.noise_std = noise_std
        self.dataset = dataset
        self.device = device

def mom_estimator(model, hparams):
    vae = model

    se = torch.nn.MSELoss(reduction='none')
    mom_batch_size = hparams.mom_batch_size
    batch_size = hparams.batch_size

    def estimator(A_val, y_val, hparams):
        """Function that returns the estimated image"""

        A = torch.Tensor(A_val).to(hparams.device)
        y = torch.Tensor(y_val).to(hparams.device)

        shuffled_idx = torch.randperm(hparams.num_measurements)


        def sample(z):
            return vae.decode(z, 1).sample

        def get_loss(xf, xg):
            # compute measurements
            if hparams.dataset=="mmnist":
                xf = 0.2989 * xf[:, 0, :, :] + 0.5870 * xf[:, 1, :, :] + 0.1140 * xf[:, 2, :, :]
            yf_batch = torch.mm(xf.view(batch_size, -1), A)
#             yg_batch = torch.mm(xg.view(batch_size, -1), A)

            # compute corresponding losses
            loss_1 = se(yf_batch, y)
#             loss_2 = se(yg_batch, y)

#             loss_3 = loss_1 - loss_2
            # now find median block of loss_1 - loss_2
            loss_3 = loss_1                               # MOM minimization, using loss1 directly

            #shuffle the losses
            loss_3 = loss_3[:,shuffled_idx]
            loss_3 = loss_3[:,:mom_batch_size*(A.shape[0]//mom_batch_size)] # make the number of rows a multiple of batch size
            loss_3 = loss_3.view(batch_size,-1,mom_batch_size) # reshape
            loss_3 = loss_3.mean(axis=-1) # find mean on each batch
            loss_3_numpy = loss_3.detach().cpu().numpy() # convert to numpy

            median_idx = np.argsort(loss_3_numpy, axis=1)[:,loss_3_numpy.shape[1]//2] # sort and pick middle element

            # pick median block
            loss_batch = loss_3[range(batch_size), median_idx] # torch.mean(loss_1_mom - loss_2_mom)
            return loss_batch


        zf_batch = torch.randn(batch_size, 4, 8, 8, dtype=torch.float32,requires_grad=True,device=hparams.device)
#         zg_batch = torch.randn(batch_size, 4, 8, 8, dtype=torch.float32,requires_grad=True,device=device)
        z_output_batch = torch.zeros(batch_size, 4, 8, 8, dtype=torch.float32,device=hparams.device)

        opt1 = optim.Adam([zf_batch], lr=hparams.learning_rate)
#         opt2 = optim.Adam([zg_batch], lr=hparams.learning_rate)

        for j in range(hparams.max_update_iter):
            xf_batch = sample(zf_batch)
            
            opt1.zero_grad()
#             xf_batch, xg_batch = sample(zf_batch), sample(zg_batch)
            xf_batch = sample(zf_batch)
            loss_f_batch = get_loss(xf_batch, xg=0)
            loss_f =  loss_f_batch.mean()
            loss_f.backward()
            opt1.step()

#             opt2.zero_grad()
#             xf_batch, xg_batch = sample(zf_batch), sample(zg_batch)
#             loss_g_batch = -1 * get_loss(xf_batch, xg_batch)
#             loss_g = loss_g_batch.mean()
#             loss_g.backward()
#             opt2.step()


            logging_format = 'iter {} loss_f {}'
            # print(logging_format.format(j, loss_f.item()))


            if j >= hparams.max_update_iter - 200:
                z_output_batch += zf_batch.detach()

        z_output_batch = z_output_batch/200
        x_hat_batch = sample(z_output_batch)
        if hparams.dataset=="mmnist":
            x_hat_batch = 0.2989 * x_hat_batch[:, 0, :, :] + 0.5870 * x_hat_batch[:, 1, :, :] + 0.1140 * x_hat_batch[:, 2, :, :]
        y_hat_batch = torch.mm(x_hat_batch.view(hparams.batch_size,-1), A)

        return x_hat_batch, z_output_batch

    return estimator