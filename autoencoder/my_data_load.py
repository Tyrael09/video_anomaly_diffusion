import os
import numpy as np

from tqdm import tqdm, trange
import pandas as pd
from PIL import Image
import imageio.v3 as iio

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# My own Dataloader. Takes a set of mp4 videos and a csv file called "header.csv" containing an ID (same as path minus ".mp4" at the end..), 
# a label (0 = regular, 1 = irregular), the total frame count and the video path, both from the same directory, applies some transformations 
# and returns the Dataset (consisting of the ID, label and tensor representations of the input frames)
class VideoDataset(Dataset):
    
    def __init__(self,
                 root_dir='/local/scratch/Cataract-1K-Hendrik/',
                 dataset_name='regular_videos_long/train/',
                 transform_fn=None):

        self.load_clip = self.load_clip_ucfc
            
        self.path = f'{root_dir}/{dataset_name}'
        self.header_df = pd.read_csv(self.path+'header.csv')
        self.lenght = len(self.header_df)
        
        im_resize = transforms.Resize((256, 256),
                                      interpolation=transforms.InterpolationMode.BICUBIC,
                                      antialias=True)
        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                         std=[0.22803, 0.22145, 0.216989])
        
        if transform_fn:
            self.transform_fn = transforms.Compose([
                im_resize,
                transforms.ToTensor(),
                transform_fn,
                normalize,
                ])
        else:
            self.transform_fn = transforms.Compose([
                im_resize,
                transforms.ToTensor(),
                normalize,
                ])
        
    
    def __len__(self):
        return self.lenght
    
    def __getitem__(self, index):
        return self.load_clip(index)
    
    
    def load_clip_ucfc(self, idx):
        video_id, label, frame_count, video_path = self.header_df.iloc[idx][['video_id', 'label', 'frame_count', 'video_path']]
        file_path = os.path.join(self.path, video_path)
        video = []
        # Read video using imageio.v3
        frames = iio.imread(file_path, plugin='pyav')

        for frame in frames:
            image = Image.fromarray(frame)
            image = self.transform_fn(image)
            video.append(image)
                
        return {'clip_id': video_id, 'label':label, 'data': torch.stack(video, dim=0)}


class VideoClipDataset(Dataset):
    def __init__(self,
                 root_dir='/local/scratch/Cataract-1K-Hendrik/regular_videos_long/',
                 dataset_name='train',
                 clip_len=16,
                 split='train',
                 load_reminder=False,
                 transform_fn=None):
        self.path = f'{root_dir}/{dataset_name}'
        self.clip_len = clip_len
        self.vid_header_df = pd.read_csv(self.path+f'/header_{split}.csv')
        if load_reminder:
            self.header_df = pd.read_csv(self.path+f'/reminders_{split}.csv')
        else:
            self.header_df = pd.read_csv(self.path+f'/header_{split}.csv') # was '/splits_header_{split}.csv'
        self.length = len(self.header_df)
        
        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                         std=[0.22803, 0.22145, 0.216989])
        
        if transform_fn:
            self.transform_fn = transforms.Compose([
                transforms.ToTensor(),
                transform_fn,
                normalize,
                ])
        else:
            self.transform_fn = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.load_clip(index)

    def load_clip(self, idx):
        video_id, start, end, label = self.header_df.iloc[idx][['video_id', 'start', 'end', 'label']]
        file_path = os.path.join(self.path, 'frames', video_id)

        clip = []
        for i in range(start, end):
            img_path = f'{file_path}/{i}.jpg'
            with Image.open(img_path) as image:
                image = self.transform_fn(image)
            clip.append(image)
        return {'clip_id': video_id, 'start': start, 'data': torch.stack(clip, dim=0)}

if __name__ == '__main__':
    print('Hello there!')
    # root_dir='../data'
    ds = VideoDataset() 
    clip = ds[0]['data']
    print(clip.shape)
    print('Obiwan Kenobi!')
