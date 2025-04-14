import yaml
from argparse import Namespace

from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW

from preprocess.video_dataset import VideoClsDataset
from models.networks import SimCLR, BYOL, MoCo, Swav
from models.trainer import train_inner, train_simclr_koop, train_byol_koop, train_moco_koop, train_swav_koop
from models.optim import LARS

def load_dataset(args):
    train_dataset = VideoClsDataset(
        args.train_root,
        clip_len=args.clip_len,
        frame_sample_rate=args.frame_sample_rate,
        args=args,
        num_segment=args.num_segment,
        crop_size=args.crop_size,
        short_side_size=args.short_side_size,
        new_height=args.new_height,
        new_width=args.new_width
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last
    )

    cls_dataset = VideoClsDataset(
        args.clf_root,
        clip_len=args.val_clip_len,
        frame_sample_rate=args.val_frame_sample_rate,
        args=args,
        mode='validation',
        crop_size=args.crop_size,
        short_side_size=args.short_side_size,
        new_height=args.new_height,
        new_width=args.new_width
    )

    val_dataset = VideoClsDataset(
        args.val_root,
        clip_len=args.val_clip_len,
        frame_sample_rate=args.val_frame_sample_rate,
        args=args,
        mode='validation',
        crop_size=args.crop_size,
        short_side_size=args.short_side_size,
        new_height=args.new_height,
        new_width=args.new_width
    )

    cls_loader = DataLoader(
        dataset=cls_dataset,
        batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    return train_loader, cls_loader, val_loader

def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)

def load_model(args):
    model_dict = {
        "simclr": SimCLR,
        "byol": BYOL,
        "swav": Swav,
        "moco": MoCo,
    }
    model = model_dict[args.model](args)
    return model

def load_optimizer(args, model):
    optimizer_dict = {
        "sgd": SGD,
        "adam": Adam,
        "adamw": AdamW,
        "lars": LARS,
    }
    optimizer_inner = optimizer_dict[args.optimizer_inner](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    optimizer_outter = optimizer_dict[args.optimizer_outter](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    return optimizer_inner, optimizer_outter

def load_functions(args):
    func_dir = {
        "simclr": train_simclr_koop,
        "byol": train_byol_koop,
        "swav": train_swav_koop,
        "moco": train_moco_koop,
    }
    inner_func = train_inner
    outter_func = func_dir[args.model]

    return inner_func, outter_func
