from omegaconf import OmegaConf
import torch
import os
from torch.utils.data import  RandomSampler, SequentialSampler
from utils import init_seeds, set_training_dir, set_log
from datasets import create_train_dataset, create_train_loader
try:
    import wandb
    deactivate_wandb = 0
except ImportError:
    print("WANDB is not available please install")
    print("use command: pip3 install wandb")
    deactivate_wandb = 1


def main(config_dict):
    scaler = torch.cuda.amp.GradScaler() if config_dict.Training.amp else None
    init_seeds(config_dict.Training.seed)
    out_dir = set_training_dir(config_dict.Training.run_name, config_dict.Training.run_tag)
    set_log(out_dir)
    OmegaConf.save(config_dict, os.path.join(out_dir, 'config.yaml'))
    print('Creating data loaders')
    train_dataset = create_train_dataset(dataset_path=config_dict.Dataset.training_path,
                                         class_name=config_dict.Dataset.class_name,
                                         img_size=config_dict.Dataset.image_size,
                                         is_train=True)
    train_sampler = RandomSampler(train_dataset,generator=torch.Generator().manual_seed(config_dict.Training.seed))
    train_dataloader = create_train_loader(train_dataset=train_dataset,
                                           batch_size=config_dict.Dataset.train_batchsize,
                                           num_workers=config_dict.Dataset.train_number_workers,
                                           batch_sampler=train_sampler)
    print('Created Data loaders')





if __name__ == "__main__":
    args = OmegaConf.load('config_baseline.yaml')
    main(args)