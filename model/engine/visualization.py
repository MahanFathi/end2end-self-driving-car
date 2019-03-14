import os
from tqdm import tqdm
import torch
from util.logger import setup_logger
from util.save_image import save_image


def worker(image, image_mask, path):
    image = image.permute(0, 3, 1, 2)
    save_image(image, path + '_image.jpg', normalize=True, padding=0)
    save_image(image_mask, path + '_backward_segmentation.jpg')


def create_path(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)


def do_visualization(
        cfg,
        model,
        backward_model,
        dataloader,
        device,
):
    logger = setup_logger('balad-mobile.visualization', False)
    logger.info("Start visualizing")

    create_path(cfg.OUTPUT.VIS_DIR)

    for iteration, (images, targets, paths) in tqdm(enumerate(dataloader)):
        indices = targets.view(-1).nonzero().type(torch.LongTensor)
        images = images.to(device)
        targets = targets.to(device)
        predictions, activations = model(images, targets)
        backward_segmentations = backward_model(activations)

        images, steering_commands, backward_segmentations, indices = \
            images.cpu(), targets.cpu(), backward_segmentations.cpu(), indices.cpu()
        for i in indices:
            image, image_mask = images[i, :, :, :], backward_segmentations[i, :, :, :]
            path = cfg.OUTPUT.VIS_DIR + "/" + paths[i]
            worker(image, image_mask, path)
