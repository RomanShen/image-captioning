from torch import nn
import torch
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, k1=0.1, k2=0.1):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.nll = F.nll_loss()
        self.k1 = k1
        self.k2 = k2

    def forward(self, y_pred, y, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = torch.mean(torch.mul(mu, mu)) + torch.mean(torch.exp(logvar)) - torch.mean(logvar) - 1
        CLS = self.nll(y_pred, y)

        return CLS + self.k1 * MSE + self.k2 * KLD