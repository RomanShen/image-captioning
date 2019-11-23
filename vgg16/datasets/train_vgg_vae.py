from torch.optim.lr_scheduler import ExponentialLR
from vgg_vae_loss import VAELoss
from models.vgg_vae import VggVAE
import torch
from torch import optim
from ignite.utils import convert_tensor


def run(lr, momentum):
    model = VggVAE()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.001)

    lr_scheduler = ExponentialLR(optimizer, gamma=0.975)

    def update_fn(engine, batch):
        model.train()

        x = convert_tensor(batch[0], device=device, non_blocking=True)
        y = convert_tensor(batch[1], device=device, non_blocking=True)

        y_pred = model(x)

        # Compute loss
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        return {
            "batchloss": loss.item(),
        }