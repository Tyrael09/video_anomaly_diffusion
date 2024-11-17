import os
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import resize
import subprocess
import cv2
import numpy as np

# Use this script to cut the videos before extracting frames and creating csv files for clips.
# TODO: figure out if the resulting frames/csv are used to feature extraction, training or evaluation.


# Function to repair the video file using ffmpeg
def repair_video(input_video_path):
    repaired_video_path = input_video_path.replace(".mp4", "_repaired.mp4")
    try:
        command = [
            "ffmpeg",
            "-i",
            input_video_path,
            "-c",
            "copy",
            "-map",
            "0",
            "-movflags",
            "faststart",
            repaired_video_path,
        ]
        subprocess.run(command, check=True)
        return repaired_video_path
    except subprocess.CalledProcessError as e:
        print(f"Error repairing video {input_video_path}: {e}")
        return None


# Function to cut video and return VideoFileClip object
def cut_video(input_video_path, start_time, end_time):
    try:
        video = VideoFileClip(input_video_path).subclip(start_time, end_time)
        video = resize(video, width=256)
        return video
    except OSError as e:
        print(f"Error processing video {input_video_path}: {e}")
        return None


# Function to extract frames from VideoFileClip object
def extract_frames(video_clip, frame_rate=2):
    frames = []
    for count, frame in enumerate(video_clip.iter_frames()):
        if count % frame_rate == 0:
            frames.append(frame)
    return frames


# Creates clips and labels for each video.
def generate_csv(frames_dict, start_times, clip_len=16):
    data = []
    # overlap = clip_len // 4  # number of frames to overlap
    for video_id, frames in frames_dict.items():
        num_clips = 5  # len(frames) // (clip_len - overlap)  # accounting for overlap
        for i in range(num_clips):
            start_frame = start_times[video_id] + i * 64
            end_frame = start_frame + clip_len * 32
            if end_frame > len(frames):  # Ensure we do not go out of bounds
                end_frame = len(frames)
                start_frame = end_frame - clip_len
                if start_frame < 0:  # Skip if we don't have enough frames
                    continue
            # Replace 'label' with your actual label logic
            label = 0  # TODO: set to 0 for regular videos, 1 for irregular videos
            data.append([video_id, start_frame, end_frame, label])
    df = pd.DataFrame(data, columns=["video_id", "start", "end", "label"])
    return df


# Cuts the videos, extracts the frames, then generates header_df to be used in
# "video_anomaly_diffusion.custom_dataset.feature_dataset.ClipDataset"
def cut_extract_dict_pipeline(
    videos_folder="/local/scratch/Cataract-1K-Full-Videos/",
    csv_file_path="/local/scratch/hendrik/video_annotations_train.csv",
    split="train",
):
    times_df = pd.read_csv(csv_file_path)
    frames_dict = {}
    start_times = {}
    # Process each row in the CSV file
    for row in times_df.itertuples(index=False):
        case = row.video_id
        start_time = row.start
        end_time = row.end
        label = row.label
        csv_split = row.split
        input_video_path = os.path.join(videos_folder, f"{case}.mp4")
        # repaired_video_path = os.path.join(output_folder, f"{case}_repaired.mp4")
        # output_path = os.path.join(output_folder, f"{case}.mp4") # fixed paths/names because "_part" is uneceesary now
        if not os.path.exists(input_video_path):
            print(f"Video file for {case} not found, skipping.")
            continue
        # Attempt to cut the video based on the provided times and get the VideoFileClip object
        if label == 0 and csv_split == split:  # label 0 means regular, split 0 means training split.
            video_clip = cut_video(input_video_path, start_time, end_time)
            # If cutting fails, attempt to repair and cut again
            if video_clip is None:
                repaired_video_path = repair_video(input_video_path)
                if repaired_video_path:
                    video_clip = cut_video(repaired_video_path, start_time, end_time)
            if video_clip is None:
                continue
            # Extract frames from the VideoFileClip object
            frames_dict[case] = extract_frames(video_clip)
            start_times[case] = int(start_time * 60)
    # Generate the DataFrame using the frames_dict
    df = generate_csv(frames_dict, start_times)
    return df
