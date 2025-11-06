import os
import sys

import numpy as np
import cv2
from moviepy import VideoFileClip

import whisperx
from pyannote.audio import Pipeline
from pyannote.core import Segment
import argparse


def main():
    """Main function to run the diarization"""   
    parser = argparse.ArgumentParser(description='Zoom Meeting Speaker Diarization')
    parser.add_argument('--video_file', help='Path to zoom video')
    parser.add_argument('--hf_token', help='HF auth token for pyannote/speaker-diarization')
    parser.add_argument('--save_dir', , help='Path to directory to save whisperDiarization')
    
    args = parser.parse_args()
    video_file_name = args.video_file
    audio_file_name = args.video_file.replace('.mp4', '.wav')
    vid_id = video_file_name.split('/')[-1].split('.')[0]

    print(f"Creating Audio File {audio_file_name}")
    model = whisperx.load_model("large-v2", device="cuda")
    result = model.transcribe(audio_file_name)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file_name, "cuda")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=args.hf_token)
    diarization = pipeline(audio_file_name, min_speakers=None, max_speakers=None)

    # Add speaker labels to WhisperX segments
    result_segments = result_aligned["segments"]  # or result["segments"] if you skip alignment

    for segment in result_segments:
        segment_start = segment["start"]
        segment_end = segment["end"]

        # Find the speaker active during this segment
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= segment_start <= turn.end or turn.start <= segment_end <= turn.end:
                segment["speaker"] = speaker
                break
            else:
                segment["speaker"] = "unknown"

    np.save(os.path.join(args.save_dir, f'{vid_id}.npy'), result_segments)


if __name__ == "__main__":
    main()

