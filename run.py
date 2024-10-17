import gradio as gr
import os
import ffmpeg
import time
import logging
from typing import Tuple
from torch import cuda
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Set up logging configuration
logging.basicConfig(filename='transcription_app_gr5.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Whisper model configuration
WHISPER_CT_MODEL_NAME = "terasut/whisper-th-large-v3-combined-ct2"
COMPUTE_TYPE = "float16"

# Check if CUDA is available and set the device accordingly
device = "cuda" if cuda.is_available() else "cpu"
logging.info(f"Loading {WHISPER_CT_MODEL_NAME} model on {device}, compute_type={COMPUTE_TYPE}")

# Load the Whisper model and create a batched inference pipeline
model = WhisperModel(WHISPER_CT_MODEL_NAME, device=device, compute_type=COMPUTE_TYPE)
batched_model = BatchedInferencePipeline(model=model)
logging.info("Model loaded successfully")

def format_time(seconds: float) -> str:
    """
    Converts time in seconds to a more readable format (H:M:S) using the time module.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Time in H:M:S format.
    """
    time_struct = time.gmtime(int(seconds))
    return time.strftime("%H:%M:%S", time_struct)

def format_transcription(segments) -> Tuple[str, str]:
    """
    Formats the transcription segments into plain text and timestamped formats.

    Args:
        segments (list): List of transcribed segments.

    Returns:
        tuple: (Plain transcription, transcription with timestamps).
    """
    output = []
    output_with_timestamps = []

    for segment in segments:
        start_time_formatted = format_time(segment.start)
        end_time_formatted = format_time(segment.end)
        
        formatted_string = f"[{start_time_formatted} -> {end_time_formatted}] {segment.text}"
        output.append(segment.text)
        output_with_timestamps.append(formatted_string)

    return "\n".join(output), "\n".join(output_with_timestamps)

def transcribe(audio_filepath):
    start_time = time.time()
    logging.info(f"Starting audio transcription for: {audio_filepath}")

    segments, info = batched_model.transcribe(audio_filepath, batch_size=16)
    plain_text, text_with_timestamps = format_transcription(segments)

    run_time = time.time() - start_time
    logging.info(f"Audio transcription completed in {run_time:.2f} seconds")

    return plain_text, text_with_timestamps, run_time

def transcribe_video(video_filepath, video_start_time, video_end_time):
    start_time = time.time()
    
    logging.info(f"Starting video transcription for: {video_filepath}")

    # Get the directory and filename of the video file
    tmp_directory = os.path.dirname(video_filepath)
    audio_filepath = os.path.join(tmp_directory, "extracted_audio.wav")

    if video_start_time == 0 and video_end_time == 0:
        logging.info("Extracting audio from the full video...")
        out, _ = (ffmpeg.input(video_filepath)
                  .output(audio_filepath, ac=1, ar='16k')
                  .overwrite_output()
                  .run(capture_stdout=True))
    else:
        logging.info(f"Extracting audio from {video_start_time} to {video_end_time} seconds...")
        out, _ = (ffmpeg.input(video_filepath, ss=video_start_time, t=video_end_time - video_start_time)
                  .output(audio_filepath, ac=1, ar='16k')
                  .overwrite_output()
                  .run(capture_stdout=True))

    logging.info("Audio extraction completed")

    plain_text, text_with_timestamps, audio_run_time = transcribe(audio_filepath)

    run_time = time.time() - start_time
    logging.info(f"Video transcription completed in {run_time:.2f} seconds")

    return plain_text, text_with_timestamps, run_time

# Gradio UI

audio_interface = gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(type="filepath")],
    outputs=[
        gr.TextArea(label="Transcript", show_copy_button=True), 
        gr.TextArea(label="Transcript with timestamps", show_copy_button=True), 
        gr.Number(label="Total run time (seconds): ", precision=2),
    ],
    title="Audio to text",
    description="Using Faster-Whisper variant of https://huggingface.co/biodatlab/whisper-th-large-v3-combined"
)

video_interface = gr.Interface(
    fn=transcribe_video,
    inputs=[gr.Video(), "number", "number"],
    outputs=[
        gr.TextArea(label="Transcript", show_copy_button=True), 
        gr.TextArea(label="Transcript with timestamps", show_copy_button=True), 
        gr.Number(label="Total run time (seconds): ", precision=2),
    ],
    title="Video to text",
    description="Using Faster-Whisper variant of https://huggingface.co/biodatlab/whisper-th-large-v3-combined"
)

demo = gr.TabbedInterface(
    [video_interface, audio_interface], 
    ["Video", "Audio"],
    "Speech to text",
)

demo.launch()