import torch
import torch.utils.data
from data import datasets


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)
    return batch_sampler


def collator(batch):
    transposed_batch = list(zip(*batch))
    images, steering_commands, paths = transposed_batch

    tensor_images = torch.stack(images)
    tensor_steering_commands = torch.stack(steering_commands)

    return tensor_images, tensor_steering_commands, paths


def make_data_loader(cfg, mode):
    """Make a dataloader.

    :param cfg: CfgNode containing the configurations of everything.
    :param mode: str in ['train', 'val', 'vis']
    :return: dataloader: torch.utils.data.DataLoader
    """
    annotation_path = {
        'train': cfg.DATASETS.ANN_TRAIN_PATH,
        'val': cfg.DATASETS.ANN_VAL_PATH,
        'vis': cfg.DATASETS.ANN_VIS_PATH,
    }[mode]
    dataset_factory = getattr(datasets, cfg.DATASETS.FACTORY)

    # Make the datasets
    dataset = dataset_factory(cfg, cfg.DATASETS.DATA_DIR, annotation_path)

    # Make the sampler
    sampler = make_data_sampler(dataset, cfg.DATASETS.SHUFFLE)

    # Make batch sampler
    batch_sampler = make_batch_data_sampler(sampler, cfg.INPUT.BATCH_SIZE)

    # Make data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )

    return data_loader
