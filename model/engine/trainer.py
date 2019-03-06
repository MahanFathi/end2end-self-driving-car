import os
import torch
from datetime import datetime
from util.logger import setup_logger
from util.visdom_plots import VisdomLogger


def do_train(
        cfg,
        model,
        dataloader,
        optimizer,
        device
):
    # set mode to training for model (matters for Dropout, BatchNorm, etc.)
    model.train()

    # get the trainer logger and visdom
    visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
    visdom.register_keys(['loss'])
    logger = setup_logger('balad-mobile.train', False)
    logger.info("Start training")

    output_dir = os.path.join(cfg.LOG.PATH, 'run_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # start the training loop
    for _ in range(cfg.SOLVER.EPOCHS):
        for iteration, (images, steering_commands) in enumerate(dataloader):
            images = images.to(device)
            steering_commands = steering_commands.to(device)

            predictions, loss = model(images, steering_commands)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % cfg.LOG.PERIOD == 0:
                visdom.update({'loss': [loss.item()]})
                logger.info("LOSS: \t{}".format(loss))

            if iteration % cfg.LOG.PLOT.ITER_PERIOD == 0:
                visdom.do_plotting()

            if iteration % cfg.LOG.WEIGHTS_SAVE_PERIOD == 1:
                torch.save(model.state_dict(), os.path.join(output_dir, 'weights_{}'.format(str(iteration))))
