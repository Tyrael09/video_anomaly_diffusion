import pandas as pd
import os
import cv2

"""
Creates clips and labels for each video. Before using this script, cut the videos using 
"video_cut.py" and extract the frames from each video with "extract_frames.py".
Example: Each video is divided into clips of 16 frames with an overlap of 4 frames with 
the previous clip. Takes in a set of frames for each video folder and outputs a csv with
all clip start/end frames and the video name
"""


def generate_csv(frames_dict, clip_len=16, overlap=4):
    data = []
    for video_id, frames in frames_dict.items():
        num_clips = len(frames) // (clip_len - overlap)  # accounting for overlap
        for i in range(num_clips):
            start_frame = i * (clip_len - overlap)
            end_frame = start_frame + clip_len
            if end_frame > len(frames):  # Ensure we do not go out of bounds
                end_frame = len(frames)
                start_frame = end_frame - clip_len
                if start_frame < 0:  # Skip if we don't have enough frames
                    continue
            # Replace 'label' with your actual label logic
            label = 0  # TODO set to 0 for regular videos, 1 for irregular videos
            data.append([video_id, start_frame, end_frame, label])
    df = pd.DataFrame(data, columns=["video_id", "start", "end", "label"])
    return df


# Example usage:
# frames_directory = '/path/to/frames_dir'
# df = generate_csv(frames_directory)
# print(df)


"""
Extracts frames from videos previously cut to the correct length, only including part after lens insertion. 
Use "video_cut_irregular.py" prior to using this script.
Later, csv files can be generated for these frames (using "generate_csv_file.py").
Change frame_rate value to skip frames. Alternatively, consider implementation like in 
feature_dataset.ClipDataset.generate_self_cleaning_indexes() to skip frames based on 
Euclidean distance.
"""


def extract_frames(video_path, frame_rate=2):
    video_frames = []
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            video_frames.append(frame)
        success, frame = video_capture.read()
        count += 1
    video_capture.release()
    return video_frames


# For external usage (in video_anomaly_diffusion.custom_datasets.feature_dataset.ClipDataset)
def extract_frames_and_create_dict(input_videos_dir="/local/scratch/Cataract-1K-Hendrik/irregular_videos/"):
    frames_dict = {}

    for video_file in os.listdir(input_videos_dir):
        video_path = os.path.join(input_videos_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        frames = extract_frames(video_path)
        frames_dict[video_id] = frames

    return generate_csv(frames_dict)


# Old implementation, do not use
"""
def generate_csv_old(frames_dir, output_csv, clip_len=16):
    data = []
    overlap = (clip_len // 4) - 1 # number of frames to overlap (maybe -1???). Depends on whether function reading from this csv file counts "end" as inclusive or not. TODO: find out (see left)
    for video_id in os.listdir(frames_dir):
        video_dir = os.path.join(frames_dir, video_id)
        if os.path.isdir(video_dir):
            frames = sorted(os.listdir(video_dir))
            num_clips = len(frames) // (clip_len - overlap) # accounting for overlap
            for i in range(num_clips + 1):
                start_frame = i * (clip_len - overlap) # added 4 frames of overlap, adapt num_clips value to reflect this
                end_frame = start_frame + clip_len
                # Replace 'label' with your actual label logic
                label = 0  # TODO: set to 0 for regular videos, 1 for irregular videos
                data.append([video_id, start_frame, end_frame, label])
    df = pd.DataFrame(data, columns=['video_id', 'start', 'end', 'label'])
    df.to_csv(output_csv, index=False)


frames_dir = '/local/scratch/Cataract-1K-Full-Videos/frames'
output_csv = '/local/scratch/Cataract-1K-Full-Videos/header_train.csv' # previously header_test.csv - TODO: check where this file is actually used in the pipeline!
generate_csv_old(frames_dir, output_csv)
"""
