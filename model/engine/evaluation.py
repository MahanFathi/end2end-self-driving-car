import numpy as np
from tqdm import tqdm
from util.logger import setup_logger


def do_evaluation(
        cfg,
        model,
        dataloader,
        device,
        verbose=False,
):
    logger = setup_logger('balad-mobile.evaluation', False)
    logger.info("Start evaluating")

    loss_records = []
    for iteration, (images, steering_commands, _) in tqdm(enumerate(dataloader)):
        images = images.to(device)
        steering_commands = steering_commands.to(device)
        predictions, loss = model(images, steering_commands)
        loss_records.append(loss.item())
        if verbose:
            logger.info("LOSS: \t{}".format(loss))

    logger.info('LOSS EVALUATION: {}'.format(np.mean(loss_records)))

