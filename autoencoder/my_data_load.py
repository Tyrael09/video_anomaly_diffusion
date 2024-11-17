import os
import cv2
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VideoClipDataset(Dataset):
    def __init__(
        self,
        root_dir="/local/scratch/Cataract-1K-Full-Videos/",
        csv="",
        clip_len=16,
        transform_fn=None,
        frame_rate=60,
        overlap=4,
        frame_skip=4,
    ):
        self.path = root_dir
        self.clip_len = clip_len
        self.frame_rate = frame_rate
        self.overlap = overlap
        self.frame_skip = frame_skip
        self.vid_header_df = pd.read_csv(csv)

        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        if transform_fn:
            self.transform_fn = transforms.Compose([transforms.ToTensor(), transform_fn, normalize])
        else:
            self.transform_fn = transforms.Compose([transforms.ToTensor(), normalize])

        # Precompute all clips and store their metadata for indexing
        self.clips = self.precompute_clips()

    def precompute_clips(self):
        clips = []
        for idx in range(len(self.vid_header_df)):
            video_id, start, end, label = self.vid_header_df.iloc[idx][["video_id", "start", "end", "label"]]
            start_frame = int(start * self.frame_rate)
            end_frame = int(end * self.frame_rate)
            video_path = os.path.join(self.path, f"{video_id}.mp4")

            frame_idx = start_frame
            while frame_idx + self.clip_len * self.frame_skip <= end_frame:
                clips.append({"video_path": video_path, "clip_start": frame_idx, "video_id": video_id})
                frame_idx += (self.clip_len - self.overlap) * self.frame_skip  # Apply overlap
        return clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip_info = self.clips[index]
        return self.load_clip(clip_info)

    def load_clip(self, clip_info):
        video_path = clip_info["video_path"]
        start_frame = clip_info["clip_start"]
        video_id = clip_info["video_id"]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        clip = []
        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise ValueError(f"Frame read error in video {video_id} at frame {start_frame}")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = self.transform_fn(image)
            clip.append(image)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + self.frame_skip - 1)

        cap.release()
        return {"clip_id": video_id, "start": start_frame, "data": torch.stack(clip, dim=0)}
