import pandas as pd
import numpy as np
import torch
import os


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir="", clip_len=16, feat_model="resnet18", split="test"):
        self.split = split
        self.clip_len = clip_len
        self.dir_name = dataset_dir
        self.feat_model = feat_model
        self.load_clip = self.load_clip_3d
        annotations = "vad_train_set.csv"
        if self.split == "test":
            annotations = "vad_test_set.csv"
        my_header = pd.read_csv(annotations)
        self.header_df = my_header[my_header["label"] != 2]
        self.length = len(self.header_df)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.load_clip(index)

    def load_clip_3d(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][["video_id", "start", "end", "label"]]
        if end == 0:
            return None
        label = np.array(label, dtype=np.int32)
        feat_path = f"/local/scratch/hendrik/vid_features_4/{video_id}/{start}.npy"
        clip = np.load(feat_path)
        return clip, label, video_id, start
