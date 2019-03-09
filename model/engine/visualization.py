import os
from tqdm import tqdm
import torch
from util.logger import setup_logger
from util.utils import save_image


def worker(image, image_mask, path):
    image = image.permute(0, 3, 1, 2)
    save_image(image, path + '_image.jpg', normalize=True, padding=0)
    save_image(image_mask, path + '_backward_segmentation.jpg')
    # image[0, 2, :] = image[0, 2, :] + 10 * image_mask[0, :, :]
    # save_image(image, path + '_visualization.jpg', normalize=True, padding=0)


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def do_visualization(
        cfg,
        model,
        backward_model,
        dataloader,
        device,
        verbose=False,
):
    logger = setup_logger('balad-mobile.visualization', False)
    logger.info("Start visualizing")

    for iteration, (images, targets, paths) in tqdm(enumerate(dataloader)):
        indices = targets.view(-1).nonzero().type(torch.LongTensor)
        images = images.to(device)
        targets = targets.to(device)
        predictions, activations = model(images, targets)
        backward_segmentaions = backward_model(activations)

        images, steering_commands, backward_segmentaions, indices = \
            images.cpu(), targets.cpu(), backward_segmentaions.cpu(), indices.cpu()
        for i in indices:
            image, image_mask = images[i, :, :, :], backward_segmentaions[i, :, :, :]
            path = cfg.OUTPUT.VIS_DIR + "/" + paths[i]
            worker(image, image_mask, path)
