from argparse import ArgumentParser
from datasets.read_images import ImageNetData
import torch
from models.vgg import Vgg16
from torch import optim
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def default_score_fn(engine):
    score = engine.state.metrics['accuracy']
    return score


def run(train_loader, val_loader, epochs, lr, momentum, log_interval, log_dir):

    model = Vgg16()

    writer = create_summary_writer(model, train_loader, log_dir)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.001)

    lr_scheduler = ExponentialLR(optimizer, gamma=0.975)

    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)

    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.nll_loss)},
                                            device=device)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # store the best model
    best_model_handler = ModelCheckpoint(
        dirname=log_dir,
        filename_prefix="best",
        n_saved=3,
        score_name="test_acc",
        score_function=default_score_fn
    )
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })

    # add early stopping
    es_patience = 5
    es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)

    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
              "".format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output))
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./dataset")
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--momentum', type=float, default=0.8,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()

    dataloaders, dataset_sizes = ImageNetData(args)

    run(dataloaders['train'], dataloaders['val'], args.epochs, args.lr, args.momentum, args.log_interval, args.log_dir)

