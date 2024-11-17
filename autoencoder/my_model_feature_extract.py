#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:32:49 2022

@author: anil
"""

import torch
from ResNext_models import resnet
import os
import argparse
import numpy as np

# from tqdm import trange, tqdm
from my_data_load import VideoClipDataset
from torch.utils.data import DataLoader


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_resnet3D(depth=50):
    model = resnet.generate_model(
        model_depth=depth,
        n_classes=1039,
        n_input_channels=3,
        shortcut_type="B",
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        widen_factor=1.0,
    )

    PATH = "autoencoder/ResNext_models/trained/r3d18_KM_200ep.pth"
    checkpoint = torch.load(PATH, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model


def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit(1)
    device = torch.cuda.current_device()
    print(f"Script is using GPU: {torch.cuda.get_device_name(device)} (ID: {device})")


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--batch_size", type=int, default=1, help="batch size")
    p.add_argument("--gpu_id", type=int, default=0, help="gpu id to choose the gpu")
    p.add_argument("--split_id", type=int, default=0, help="split id to choose the split")
    p.add_argument("--split_n", type=int, default=1, help="n split to divide")
    p.add_argument("--split_name", type=str, default="train", help="dataset split to process")
    p.add_argument("--reminders", action="store_true", default=False, help="reminder switch")
    return p.parse_args()


if __name__ == "__main__":
    # from time import time
    print("Hello there!")
    args = args_parser()

    gpu_id = args.gpu_id
    split_id = args.split_id
    split_n = args.split_n
    clip_len = 16
    split_name = args.split_name
    reminders = args.reminders
    depth = 18
    feat_model = f"r3D{depth}"
    model = load_resnet3D(depth)
    # feat_model = 'rx3D'; model = load_resnext3D()
    batch_size = args.batch_size

    model = model.eval()
    model = model.cuda(gpu_id)

    with torch.no_grad():
        root_dir = "/local/scratch/Cataract-1K-Full-Videos/"
        segment_dir = "/local/scratch/Cataract-1K/Phase_recognition_dataset/videos/"

        dst_data_path = f"/local/scratch/hendrik/vid_features_4"

        my_csv = "/local/scratch/hendrik/regular_video_annotations_full.csv"
        segment_an = "/local/scratch/hendrik/segmentation_annotations.csv"

        ds = VideoClipDataset(root_dir=segment_dir, csv=segment_an)
        mkdir(dst_data_path)

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            prefetch_factor=2,
        )

        for batch in loader:
            clip_ids = batch["clip_id"]
            starts = batch["start"]
            x = batch["data"]

            x = x.permute((0, 2, 1, 3, 4))
            x = x.cuda(gpu_id)
            outputs = model(x).cpu()

            for clip_id, start, out in zip(clip_ids, starts, outputs):
                path = f"{dst_data_path}/{clip_id}"
                mkdir(path)
                np.save(f"{path}/{start}.npy", out)
    print("Obiwan Kenobi!")
