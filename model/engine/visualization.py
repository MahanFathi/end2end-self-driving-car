import os
from tqdm import tqdm
import torch
from util.logger import setup_logger
from util.utils import save_image
import multiprocessing as ml


def pooling_worker(image, image_mask, path):
    save_path = os.path.join(path + "/res_")
    create_path(os.path.join(save_path.split('/')[0], save_path.split('/')[1]))
    save_image(image, save_path + '_image.png', normalize=True, padding=0)
    save_image(image_mask, save_path + '_backward_segmentation.jpg')
    image[0, :, :] = image[0, :, :] + 10 * image_mask[0, :, :]
    save_image(image, save_path + '_visualization.jpg', normalize=True, padding=0)


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

    for iteration, (images, steering_commands) in tqdm(enumerate(dataloader)):
        indices = steering_commands.view(-1).nonzero().type(torch.LongTensor)
        images = images.to(device)
        steering_commands = steering_commands.to(device)
        predictions, activations = model(images, steering_commands)
        backward_segmentaions = backward_model(activations)

        # pool = ml.Pool(processes=1)
        images, steering_commands, backward_segmentaions, indices = \
            images.cpu(), steering_commands.cpu(), backward_segmentaions.cpu(), indices.cpu()
        for i in indices:
            image, image_mask, path = images[i, :, :, :], backward_segmentaions[i, :, :, :], cfg.OUTPUT.VIS_DIR
            # pool.apply_async(pooling_worker, args=(image, image_mask, path))
            pooling_worker(image, image_mask, path)

        # pool.close()
        # pool.join()
