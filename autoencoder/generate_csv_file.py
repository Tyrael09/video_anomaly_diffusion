import pandas as pd
import os

# Define your clips and labels
# Example: Each video is divided into clips of 16 frames
def generate_csv(frames_dir, output_csv, clip_len=16):
    data = []
    for video_id in os.listdir(frames_dir):
        video_dir = os.path.join(frames_dir, video_id)
        if os.path.isdir(video_dir):
            frames = sorted(os.listdir(video_dir))
            num_clips = len(frames) // clip_len
            for i in range(num_clips):
                start_frame = i * clip_len
                end_frame = start_frame + clip_len
                # Replace 'label' with your actual label logic
                label = 0  
                data.append([video_id, start_frame, end_frame, label])
    df = pd.DataFrame(data, columns=['video_id', 'start', 'end', 'label'])
    df.to_csv(output_csv, index=False)


frames_dir = '/local/scratch/Cataract-1K-Hendrik/irregular_videos/frames/'
output_csv = '/local/scratch/Cataract-1K-Hendrik/irregular_videos/frames/header_test.csv'
generate_csv(frames_dir, output_csv)
