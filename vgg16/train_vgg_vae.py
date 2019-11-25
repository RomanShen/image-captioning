import logging
import argparse
from functools import partial
from datetime import datetime

from vgg_vae_loss import VAELoss
from models.vgg_vae import VggVAE
from datasets.read_images import ImageNetData

from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import optim

from ignite.utils import convert_tensor
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy, Loss
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.contrib.metrics.gpu_info import GpuInfo


# Setup engine &  logger
def setup_logger(logger):
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _prepare_batch(batch, device, non_blocking):
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def run(train_loader, val_loader, epochs, lr, momentum, weight_decay, lr_step, k1, k2, es_patience, log_dir):
    model = VggVAE()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model.to(device)

    # SGD and lr_scheduler for SGD
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # lr_scheduler = ExponentialLR(optimizer, gamma=0.975)

    # Adam and lr_scheduler for Adam
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=2)

    criterion = VAELoss(k1=k1, k2=k2).to(device)

    def update_fn(engine, batch):
        x, y = _prepare_batch(batch, device=device, non_blocking=True)

        model.train()

        optimizer.zero_grad()

        output, x_recon, mu, logvar = model(x)

        # Compute loss
        loss = criterion(output, y, x_recon, x, mu, logvar)

        loss.backward()

        optimizer.step()

        return {
            "batchloss": loss.item(),
        }
    trainer = Engine(update_fn)

    try:
        GpuInfo().attach(trainer)
    except RuntimeError:
        print("INFO: By default, in this example it is possible to log GPU information (used memory, utilization). "
              "As there is no pynvml python package installed, GPU information won't be logged. Otherwise, please "
              "install it : `pip install pynvml`")
    # handler for SGD, if use this, comment handler for Adam in line 152
    # trainer.add_event_handler(Events.ITERATION_COMPLETED(every=lr_step), lambda engine: lr_scheduler.step())

    metric_names = [
        'batchloss',
    ]

    def output_transform(x, name):
        return x[name]

    for n in metric_names:
        # We compute running average values on the output (batch loss) across all devices
        RunningAverage(output_transform=partial(output_transform, name=n),
                       epoch_bound=False, device=device).attach(trainer, n)

    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir + "/vgg_vae/{}".format(exp_name)

    tb_logger = TensorboardLogger(log_dir=log_path)

    tb_logger.attach(trainer,
                     log_handler=OutputHandler(tag="training",
                                               metric_names=metric_names),
                     event_name=Events.ITERATION_COMPLETED)

    tb_logger.attach(trainer,
                     log_handler=OptimizerParamsHandler(optimizer, "lr"),
                     event_name=Events.ITERATION_STARTED)

    ProgressBar(persist=True, bar_format="").attach(trainer,
                                                    event_name=Events.EPOCH_STARTED,
                                                    closing_event_name=Events.COMPLETED)
    ProgressBar(persist=False, bar_format="").attach(trainer, metric_names=metric_names)

    # val process definition
    def loss_output_transform(output):
        return output[0], output[1], {'recon_x': output[2], 'x': output[3], 'mu': output[4], 'logvar': output[5]}

    def acc_output_transform(output):
        y_pred, y = output[0], output[1]
        return y_pred, y

    customed_loss = Loss(loss_fn=criterion, output_transform=loss_output_transform, device=device)
    customed_accuracy = Accuracy(output_transform=acc_output_transform, device=device)

    metrics = {
        'Loss': customed_loss,
        'Accuracy': customed_accuracy
    }

    def val_update_fn(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = _prepare_batch(batch, device=device, non_blocking=True)
            output, x_recon, mu, logvar = model(x)
            return output, y, x_recon, x, mu, logvar

    val_evaluator = Engine(val_update_fn)
    # train_evaluator = Engine(val_update_fn)

    for name, metric in metrics.items():
        # metric.attach(train_evaluator, name)
        metric.attach(val_evaluator, name)

    def run_evaluation(engine):
        # train_evaluator.run(train_loader)
        val_evaluator.run(val_loader)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluation)
    trainer.add_event_handler(Events.COMPLETED, run_evaluation)

    # handler for Adam, if use this, comment handler for SGD in line 85
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step(
        val_evaluator.state.metrics['Loss'])
    )

    # show training metrics at the end of the progress bar
    # ProgressBar(persist=False, desc="Train evaluation").attach(train_evaluator, metric_names=list(metric.keys()))
    # show validation metrics at the end of the progress bar
    ProgressBar(persist=False, desc="Validation evaluation").attach(val_evaluator)

    # Log val metrics:
    tb_logger.attach(val_evaluator,
                     log_handler=OutputHandler(tag="validation",
                                               metric_names=list(metrics.keys()),
                                               another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    # tb_logger.attach(train_evaluator,
    #                  log_handler=OutputHandler(tag="training",
    #                                            metric_names=list(metrics.keys()),
    #                                            another_engine=trainer),
    #                  event_name=Events.EPOCH_COMPLETED)

    # trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # Store the best model
    def default_score_fn(engine):
        score = engine.state.metrics['Accuracy']
        return score

    best_model_handler = ModelCheckpoint(dirname=log_path,
                                         filename_prefix="best",
                                         n_saved=3,
                                         score_name="val_acc",
                                         score_function=default_score_fn)
    val_evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })

    # Add early stopping
    es_patience = es_patience
    es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, es_handler)

    setup_logger(es_handler._logger)
    setup_logger(logging.getLogger("ignite.engine.engine.Engine"))

    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    val_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
    # train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training Vgg16 with VAE on NWPU-RESISC45 dataset")

    parser.add_argument('--data_dir', type=str, default="./dataset",
                        help="specify the path of the dataset")
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--lr_step', type=int, default=250)
    parser.add_argument('--k1', type=float, default=0.1,
                        help="weight of MSE loss ")
    parser.add_argument('--k2', type=float, default=0.1,
                        help="weight of KL loss")
    parser.add_argument('--es_patience', type=int, default=5,
                        help='how many batches to wait before early stopping')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    dataloaders, dataset_sizes = ImageNetData(args)

    run(dataloaders['train'], dataloaders['val'], args.epochs, args.lr, args.momentum, args.weight_decay, args.lr_step,
        args.k1, args.k2, args.es_patience, args.log_dir)
