import cv2
import os

# Used to extract frames from cut videos. Later, csv files can be generated for these frames.

def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"{count}.jpg")
            cv2.imwrite(frame_filename, frame)
        success, frame = video_capture.read()
        count += 1
    video_capture.release()

input_videos_dir = '/local/scratch/Cataract-1K-Hendrik/irregular_videos/'
output_frames_dir = '/local/scratch/Cataract-1K-Hendrik/irregular_videos/frames/'

for video_file in os.listdir(input_videos_dir):
    video_path = os.path.join(input_videos_dir, video_file)
    video_id = os.path.splitext(video_file)[0]
    output_dir = os.path.join(output_frames_dir, video_id)
    extract_frames(video_path, output_dir)
