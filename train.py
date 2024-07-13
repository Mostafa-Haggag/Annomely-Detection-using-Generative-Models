from omegaconf import OmegaConf
import os
import io
from PIL import Image
from tqdm import tqdm
# from Models.create_vae_model import create_model
from Models import *
from torch.utils.data import RandomSampler, SequentialSampler
from utils import init_seeds, set_training_dir, set_log, Averager
from datasets import create_train_dataset, create_train_loader, normal_params
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torchvision import transforms as T

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
try:
    import wandb
    deactivate_wandb = 0
except ImportError:
    print("WANDB is not available please install")
    print("use command: pip3 install wandb")
    deactivate_wandb = 1


def get_inv_trans(class_name):
    if class_name not in normal_params:
        raise ValueError(f"Class '{class_name}' not found in class_transforms dictionary")

    mean = normal_params[class_name]['mean']
    std = normal_params[class_name]['std']

    return T.Compose([
        T.Normalize(mean=[0.], std=[1 / s for s in std]),
        T.Normalize(mean=[-m for m in mean], std=[1.]),
    ])


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def train(model, train_loader, device, optimizer, epoch,kld_weight,class_dataset):
    model.train()
    train_loss = 0
    train_kld = 0
    train_mse = 0
    train_size = 0
    wandb_images_images = []
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for batch_idx, data in enumerate(train_loader):
            data_input,_,_ = data
            data_input = data_input.to(device)
            optimizer.zero_grad()
            results = model(data_input)
            total_loss = model.loss_function(*results,kld_weight=kld_weight)
            train_kld += total_loss["KLD"].item()
            train_mse += total_loss["Reconstruction_Loss"].item()
            total_loss["loss"].backward()
            train_loss += total_loss["loss"].item()
            train_size += data_input.shape[0]
            optimizer.step()
            pbar.set_postfix({"Total Loss": total_loss["loss"].item()})
            pbar.update()
            if (batch_idx + 1) == len(train_loader) :
                org_data = get_inv_trans(class_dataset)(data_input[:5])
                reconst_data = get_inv_trans(class_dataset)(results[0][:5])
                original_images_plotting = org_data.detach().cpu().numpy()
                reconstructed_images_plotting = reconst_data.detach().cpu().numpy()
                for i, (original, reconstructed) in enumerate(zip(original_images_plotting,
                                                                  reconstructed_images_plotting)):
                    original = (original*255).astype(np.uint8).transpose(1, 2, 0)
                    reconstructed = (reconstructed*255).astype(np.uint8).transpose(1, 2, 0)
                    fig, (
                        orginal_img,
                        reconstructed_img,
                    ) = plt.subplots(
                        nrows=1,
                        ncols=2,
                        figsize=((original.shape[1] * 2) / 96, original.shape[0] / 96),
                        dpi=96,
                    )
                    orginal_img.axis("off")
                    orginal_img.imshow(original, cmap='gray')
                    orginal_img.set_title("Org", fontsize=12)
                    reconstructed_img.axis("off")
                    reconstructed_img.imshow( reconstructed, cmap='gray')
                    reconstructed_img.set_title("Reconst", fontsize=12)
                    final_image = fig2img(fig)
                    plt.close(fig)
                    plt.close("all")
                    wandb_images_images.append(wandb.Image(final_image))
    train_loss /= train_size
    train_mse /= train_size
    train_kld /= train_size
    wandb.log({"train/Paired Images": wandb_images_images,"train/Total Loss": train_loss,
                "train/KLD":train_kld,"train/mse":train_mse}, step=epoch)

    return train_loss


def print_parameters_per_module(model: nn.Module):
    for attr in dir(model):
        module = getattr(model, attr)
        if isinstance(module, nn.Module):
            num_params = sum(p.numel() for p in module.parameters())
            print(f"Module: {attr}, Number of parameters: {num_params}")


def main(config_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"This code is running on :{device}")
    args_dict = OmegaConf.to_container(config_dict, resolve=True)
    # WANBD part

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_name = f'{timestamp}_{config_dict.Training.run_name}_{config_dict.Training.run_tag}'
    run_id = wandb.util.generate_id()
    print(f"Starting a new wandb run with id {run_id}")
    wandb.init(
            # set the wandb project where this run will be logged
            project="vae_AD",
            config=args_dict,
            tags=["AD", config_dict.Training.run_tag],
        )
    wandb.run.name = new_run_name
    scaler = torch.cuda.amp.GradScaler() if config_dict.Training.amp else None
    init_seeds(config_dict.Training.seed)
    out_dir = set_training_dir(config_dict.Training.run_name)
    set_log(out_dir)
    OmegaConf.save(config_dict, os.path.join(out_dir, 'config.yaml'))
    print('Creating data loaders')
    train_dataset = create_train_dataset(dataset_path=config_dict.Dataset.training_path,
                                         class_name=config_dict.Dataset.class_name,
                                         img_size=config_dict.Model.image_size,
                                         is_train=True)
    train_sampler = RandomSampler(train_dataset,generator=torch.Generator().manual_seed(config_dict.Training.seed))
    train_dataloader = create_train_loader(train_dataset=train_dataset,
                                           batch_size=config_dict.Dataset.train_batchsize,
                                           num_workers=config_dict.Dataset.train_number_workers,
                                           batch_sampler=train_sampler)
    print('Created Data loaders')
    print(f"Number of training samples: {len(train_dataset)}")
    model_paramaters = {"in_channels": config_dict.Model.in_channels,
                        "latent_dim": config_dict.Model.latent_dim,
                        "hidden_dims": config_dict.Model.hidden_dim,
                        "image_size": config_dict.Model.image_size}
    print(model_paramaters)
    model = create_model[config_dict.Model.type](**model_paramaters).to(device)
    if config_dict.Training.wandb_watch:
        wandb.watch(model, criterion='gradients', log_freq=100)
    print_parameters_per_module(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')
    # print(config_dict.Model.type)
    # print(create_model[config_dict.Model.type])
    # model = create_model[config_dict.Model.type](**model_paramaters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict.Training.lr)
    pbar = tqdm(range(1, config_dict.Training.num_epochs))
    for epoch in pbar:
        train(model, train_dataloader, device, optimizer,
              epoch,
              config_dict.Training.kl_weight,
              config_dict.Dataset.class_name
              )


if __name__ == "__main__":
    args = OmegaConf.load('Config/vae.yaml')
    main(args)
    wandb.finish()