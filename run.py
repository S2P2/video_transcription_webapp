import gradio as gr
import os
import ffmpeg
import time
import logging
import tomlkit
from typing import List, Tuple
from torch import cuda
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Set up logging configuration
logging.basicConfig(filename='./mount/transcription_app.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

try:
    config = tomlkit.loads(open("./mount/config.toml").read())
    logging.info("Loading config from config.toml")

except Exception as e:
    logging.error(f"Error loading config: {str(e)}")
    raise RuntimeError("Failed to load config.") from e

# Whisper model configuration
WHISPER_CT_MODEL_NAME = config["model"]["name"]
COMPUTE_TYPE = config["model"]["compute_type"]

# Check if CUDA is available and set the device accordingly
device = "cuda" if cuda.is_available() else "cpu"
logging.info(f"Loading {WHISPER_CT_MODEL_NAME} model on {device}, compute_type={COMPUTE_TYPE}")

try:
    # Load the Whisper model and create a batched inference pipeline
    model = WhisperModel(WHISPER_CT_MODEL_NAME, device=device, compute_type=COMPUTE_TYPE)
    batched_model = BatchedInferencePipeline(model=model)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load Whisper model.") from e

def format_time(seconds: float) -> str:
    """
    Converts time in seconds to a more readable format (H:M:S) using the time module.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Time in H:M:S format.
    """
    try:
        time_struct = time.gmtime(int(seconds))
        return time.strftime("%H:%M:%S", time_struct)
    except Exception as e:
        logging.error(f"Error formatting time: {str(e)}")
        raise

def format_transcription(segments: List, offset: float = 0.0) -> Tuple[str, str]:
    """
    Formats the transcription segments into plain text and timestamped formats, with an optional offset applied to timestamps.

    Args:
        segments (list): List of transcribed segments.
        offset (float, optional): Time in seconds to shift the timestamps to account for video trimming. Defaults to 0.

    Returns:
        tuple: (Plain transcription, transcription with timestamps).
    """
    output = []
    output_with_timestamps = []

    try:
        for segment in segments:
            # Apply the offset to adjust the timestamps
            start_time_shifted = segment.start + offset
            end_time_shifted = segment.end + offset

            start_time_formatted = format_time(start_time_shifted)
            end_time_formatted = format_time(end_time_shifted)

            formatted_string = f"[{start_time_formatted} -> {end_time_formatted}] {segment.text}"
            output.append(segment.text)
            output_with_timestamps.append(formatted_string)

        return "\n".join(output), "\n".join(output_with_timestamps)
    except Exception as e:
        logging.error(f"Error formatting transcription: {str(e)}")
        raise

def transcribe(audio_filepath: str) -> Tuple[str, str, float]:
    """
    Transcribes audio to text using the Faster Whisper model.

    Args:
        audio_filepath (str): Path to the audio file.

    Returns:
        tuple: (Plain transcription, transcription with timestamps, run time).
    """
    try:
        start_time = time.time()
        logging.info(f"Starting audio transcription for: {audio_filepath}")

        segments, info = batched_model.transcribe(audio_filepath, batch_size=16)
        plain_text, text_with_timestamps = format_transcription(segments)

        run_time = time.time() - start_time
        logging.info(f"Audio transcription completed in {run_time:.2f} seconds")

        return plain_text, text_with_timestamps, run_time
    except FileNotFoundError as e:
        logging.error(f"Audio file not found: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error during audio transcription: {str(e)}")
        raise

def transcribe_video(video_filepath: str, video_start_time: float, video_end_time: float) -> Tuple[str, str, float]:
    """
    Transcribes video to text by first extracting audio and then using the Faster Whisper model.

    Args:
        video_filepath (str): Path to the video file.
        video_start_time (float): Start time for the video clip to transcribe.
        video_end_time (float): End time for the video clip to transcribe.

    Returns:
        tuple: (Plain transcription, transcription with timestamps, run time).
    """
    try:
        start_time = time.time()
        logging.info(f"Starting video transcription for: {video_filepath}")

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

        segments, info = batched_model.transcribe(audio_filepath, batch_size=16)
        plain_text, text_with_timestamps = format_transcription(segments, offset=video_start_time)

        run_time = time.time() - start_time
        logging.info(f"Video transcription completed in {run_time:.2f} seconds")

        return plain_text, text_with_timestamps, run_time
    except FileNotFoundError as e:
        logging.error(f"Video file not found: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error during video transcription: {str(e)}")
        raise

# Gradio UI

audio_interface = gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(label="Audio", type="filepath")],
    outputs=[
        gr.TextArea(label="Transcript", show_label=True, show_copy_button=True), 
        gr.TextArea(label="Transcript with timestamps", show_label=True, show_copy_button=True), 
        gr.Number(label="Total run time (seconds): ", precision=2),
    ],
    description="Using Faster-Whisper variant of https://huggingface.co/biodatlab/whisper-th-large-v3-combined to transcribe audio"
)

video_interface = gr.Interface(
    fn=transcribe_video,
    inputs=[
        gr.Video(label="Video"), 
        gr.Number(label="Start Time for Transcription (seconds)"), 
        gr.Number(label="End Time for Transcription (seconds)")
    ],
    outputs=[
        gr.TextArea(label="Transcript", show_label=True, show_copy_button=True), 
        gr.TextArea(label="Transcript with timestamps", show_label=True, show_copy_button=True), 
        gr.Number(label="Total run time (seconds): ", precision=2),
    ],
    description="Using ffmpeg to extract audio and Faster-Whisper variant of https://huggingface.co/biodatlab/whisper-th-large-v3-combined to transcribe audio"
)

demo = gr.TabbedInterface(
    [video_interface, audio_interface], 
    ["Video", "Audio"],
    "แปลงเสียงเป็นข้อความ",
)

demo.launch()