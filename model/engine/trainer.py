import os
import torch
from model.engine.evaluation import do_evaluation
from datetime import datetime
from util.logger import setup_logger
from util.visdom_plots import VisdomLogger


def do_train(
        cfg,
        model,
        dataloader_train,
        dataloader_evaluation,
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
    os.makedirs(output_dir)

    # start the training loop
    for epoch in range(cfg.SOLVER.EPOCHS):
        for iteration, (images, steering_commands, _) in enumerate(dataloader_train):
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

            step = epoch * len(dataloader_train) + iteration
            if step % cfg.LOG.WEIGHTS_SAVE_PERIOD == 0 and iteration:
                torch.save(model.state_dict(),
                           os.path.join(output_dir, 'weights_{}.pth'.format(str(step))))
                do_evaluation(cfg, model, dataloader_evaluation, device)

    torch.save(model.state_dict(), os.path.join(output_dir, 'weights_final.pth'))
