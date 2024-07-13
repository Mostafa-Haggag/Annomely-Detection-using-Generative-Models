import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from utils import calculate_mean_std
from torch.utils import data

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

normal_params= {'screw':{"mean":[0.7222489502266376],"std":[0.13370024503095784]}}


# Taken from https://github.com/tkdleksms/LG_ES_anomaly/blob/main/AnoViT/dataset.py

class MVTecDataset(Dataset):
    def __init__(self, image_size, dataset_path, class_name, is_train=True, transform_x=T.ToTensor()):
        super(MVTecDataset, self).__init__()
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.transform_x = transform_x

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_mask = T.Compose([T.Resize(self.image_size, T.InterpolationMode.BILINEAR),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # are you sure you want to convert it to rgb ??? It is laoded Grey scale
        x = Image.open(x)
        x = self.transform_x(x)

        if mask == None:
            mask = torch.zeros([1, self.image_size, self.image_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []
        # phase is train or test
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        # you have the full directory until
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        img_types = sorted(os.listdir(img_dir))
        # if train --> only good
        # if test --> there is good or defects

        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])# load all pictures in this folder
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)


def get_train_transform(img_dir,class_name,img_size):
    # mean,std = calculate_mean_std(os.path.join(img_dir, class_name, 'train', 'good'))
    # transform.append(T.RandomCrop((128,128)))
    # transform.append(T.RandomHorizontalFlip(p=0.5))
    # transform.append(T.RandomVerticalFlip(p=0.5))

    return T.Compose([
                      T.Resize((img_size, img_size)),
                      T.RandomHorizontalFlip(p=0.5),
                      T.RandomVerticalFlip(p=0.5),
                      T.ToTensor(),
                     T.Normalize(mean=normal_params[class_name]["mean"],std=normal_params[class_name]["std"]),
                    ])


def create_train_dataset(
    dataset_path,
    class_name,
    img_size,
    is_train=False,
):
    train_dataset = MVTecDataset(
        img_size,
        dataset_path,
        class_name,
        is_train,
        get_train_transform(dataset_path,class_name,img_size),
    )

    # train_sampler = RandomSampler(train_dataset,generator=torch.Generator().manual_seed(seed))
    # that why we donot care about fucking
    # valid_sampler = SequentialSampler(valid_dataset)
    return train_dataset


def create_test_dataset(
    dataset_path,
    class_name,
    img_size,
    is_train=False,
):
    valid_dataset = MVTecDataset(
        img_size,
        dataset_path,
        class_name,
        is_train,
        get_train_transform(dataset_path,class_name,img_size),
    )
    # valid_sampler = SequentialSampler(valid_dataset,generator=torch.Generator().manual_seed(seed))
    return valid_dataset


def create_train_loader(
    train_dataset, batch_size, num_workers=0, batch_sampler=None
):
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=batch_sampler
    )
    return train_loader


def create_test_loader(
    test_dataset, batch_size, num_workers=0, batch_sampler=None
):
    valid_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=batch_sampler
    )
    return valid_loader

