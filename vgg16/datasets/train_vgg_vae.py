from torch.optim.lr_scheduler import ExponentialLR
from vgg_vae_loss import VAELoss
from models.vgg_vae import VggVAE
import torch
from torch import optim
from ignite.utils import convert_tensor
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import RunningAverage, Accuracy, Loss

from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
from datetime import datetime
from ignite.contrib.handlers import ProgressBar
import logging
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan


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


def run(train_loader, val_loader, epochs, lr, momentum, weight_decay, k1, k2, es_patience, log_dir):
    model = VggVAE()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = ExponentialLR(optimizer, gamma=0.975)

    criterion = VAELoss(k1=k1, k2=k2).to(device)

    def update_fn(engine, batch):
        x, y = _prepare_batch(batch, device=device, non_blocking=True)

        model.train()

        output, x_recon, mu, logvar = model(x)

        # Compute loss
        loss = criterion(output, y, x_recon, x, mu, logvar)

        optimizer.zero_grad()

        optimizer.step()

        return {
            "batchloss": loss.item(),
        }
    trainer = Engine(update_fn)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=250), lambda engine: lr_scheduler.step())

    def output_transform(out):
        return out['batchloss']
    RunningAverage(output_transform=output_transform, device=device).attach(trainer, "batchloss")

    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir + "/vgg_vae/{}".format(exp_name)
    tb_logger = TensorboardLogger(log_dir=log_path)
    tb_logger.attach(trainer,
                     log_handler=OutputHandler('training', ['batchloss', ]),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=OptimizerParamsHandler(optimizer, "lr"),
                     event_name=Events.ITERATION_STARTED)

    ProgressBar(persist=True, bar_format="").attach(trainer,
                                                    event_name=Events.EPOCH_STARTED,
                                                    closing_event_name=Events.COMPLETED)

    metrics = {
        'Loss': Loss(criterion, device=device),
        'Accuracy': Accuracy(device=device),
    }

    def val_update_fn(engine, batch):
        x, y = _prepare_batch(batch, device=device, non_blocking=True)

        model.eval()
        with torch.no_grad():
            output, x_recon, mu, logvar = model(x)

        # Compute loss
        loss = criterion(output, y, x_recon, x, mu, logvar)

        return {
            "batchloss": loss.item(),
        }

    val_evaluator = Engine(val_update_fn)
    for name, metric in metrics.items():
        metric.attach(val_evaluator, name)

    def run_evaluation(engine):
        val_evaluator.run(val_loader)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluation)
    trainer.add_event_handler(Events.COMPLETED, run_evaluation)

    # Log val metrics:
    tb_logger.attach(val_evaluator,
                     log_handler=OutputHandler(tag="val",
                                               metric_names=list(metrics.keys()),
                                               another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

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

    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    val_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    trainer.run(train_loader, max_epochs=epochs)

