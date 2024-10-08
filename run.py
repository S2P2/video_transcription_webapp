import gradio as gr
import os
import ffmpeg
import time
import logging
from typing import Tuple
from torch import cuda
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
logging.basicConfig(filename='transcription_app.log', level=logging.INFO, 
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

def check_file_exists(filepath: str) -> bool:
    """
    Checks if the provided file exists.

    Args:
        filepath (str): The path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return False
    return True

def create_tmp_folder() -> None:
    """
    Ensures that the temporary folder exists.
    """
    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')
        logging.info("Temporary folder created at ./tmp/")

def save_transcription_to_file(filepath: str, content: str) -> None:
    """
    Saves the transcription content to a file.

    Args:
        filepath (str): The path where the file will be saved.
        content (str): The content to be saved.
    """
    with open(filepath, "w") as text_file:
        text_file.write(content)
    logging.info(f"Transcription saved to {filepath}")

def format_time(seconds: float) -> str:
    """
    Converts time in seconds to a more readable format (H:M:S) using the time module.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Time in H:M:S format.
    """
    time_struct = time.gmtime(int(seconds))  # Convert to time structure
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
        # Use the updated format_time function to format the start and end times
        start_time_formatted = format_time(segment.start)
        end_time_formatted = format_time(segment.end)
        
        formatted_string = f"[{start_time_formatted} -> {end_time_formatted}] {segment.text}"
        output.append(segment.text)
        output_with_timestamps.append(formatted_string)

    return "\n".join(output), "\n".join(output_with_timestamps)

def transcribe_audio(audio_filepath: str) -> Tuple[str, str, float]:
    """
    Transcribes the given audio file using the Whisper model.

    Args:
        audio_filepath (str): Path to the audio file to be transcribed.

    Returns:
        tuple: A tuple containing the transcription (str), transcription with timestamps (str),
               and the time taken for transcription (float).
    """
    if not check_file_exists(audio_filepath):
        return "Error: Audio file not found.", "", 0
    
    try:
        start_time = time.time()
        logging.info(f"Starting audio transcription for: {audio_filepath}")

        # Prepare for audio transcription
        segments, info = batched_model.transcribe(audio_filepath, batch_size=16)
        logging.info("Audio transcribe - Done.\nDetected language '%s' with probability %f" % (info.language, info.language_probability))

        # Transcribe the audio
        plain_text, text_with_timestamps = format_transcription(segments)

        # Create the tmp directory and save files
        create_tmp_folder()
        save_transcription_to_file("./tmp/transcript.txt", plain_text)
        save_transcription_to_file("./tmp/transcript_with_timestamps.txt", text_with_timestamps)

        run_time = time.time() - start_time
        logging.info(f"Audio transcription completed in {run_time:.2f} seconds")

        return plain_text, text_with_timestamps, run_time
    
    except Exception as e:
        logging.error(f"Error during audio transcription: {str(e)}")
        return f"Error occured: {str(e)}", "", 0

def extract_audio_from_video(video_filepath: str, start_time: float, end_time: float) -> str:
    """
    Extracts audio from the video file using ffmpeg.

    Args:
        video_filepath (str): Path to the video file.
        start_time (float): Start time for extraction.
        end_time (float): End time for extraction.

    Returns:
        str: Path to the extracted audio file.
    """
    logging.info("Extracting audio from video...")

    audio_output = './tmp/tmp_audio.wav'
    if start_time == 0 and end_time == 0:
        logging.info("Extracting audio from the full video...")
        out, _ = (ffmpeg.input(video_filepath)
                  .output(audio_output, ac=1, ar='16k')
                  .overwrite_output()
                  .run(capture_stdout=True))
    else:
        logging.info(f"Extracting audio from {start_time} to {end_time} seconds...")
        out, _ = (ffmpeg.input(video_filepath, ss=start_time, t=end_time - start_time)
                  .output(audio_output, ac=1, ar='16k')
                  .overwrite_output()
                  .run(capture_stdout=True))

    logging.info("Audio extraction completed")
    return audio_output

def transcribe_video(video_filepath: str, video_start_time: float, video_end_time: float) -> Tuple[str, str, float]:
    """
    Extracts audio from the video file, transcribes the audio, and returns the transcription.

    Args:
        video_filepath (str): Path to the video file.
        video_start_time (float): Start time (in seconds) to extract audio from the video.
        video_end_time (float): End time (in seconds) to extract audio from the video.

    Returns:
        tuple: A tuple containing the transcription (str), transcription with timestamps (str),
               and the total time taken (float).
    """
    if not check_file_exists(video_filepath):
        return "Error: Video file not found.", "", 0
    
    try:
        start_time = time.time()
        logging.info(f"Starting video transcription for: {video_filepath}")

        # Extract audio from video
        create_tmp_folder()
        audio_filepath = extract_audio_from_video(video_filepath, video_start_time, video_end_time)

        # Transcribe the extracted audio
        output, output_with_timestamps, audio_run_time = transcribe_audio(audio_filepath)

        run_time = time.time() - start_time
        logging.info(f"Video transcription completed in {run_time:.2f} seconds")
        
        return output, output_with_timestamps, run_time

    except Exception as e:
        logging.error(f"Error during video transcription: {str(e)}")
        return f"Error occurred: {str(e)}", "", 0


# Gradio UI to create an interactive transcription app
with gr.Blocks() as demo:
    gr.Markdown("# Video/Audio Transcription app")

    # Layout with two columns: Video and Audio transcription
    with gr.Row():
        with gr.Column():
            video_input = gr.Video()
            video_start_time = gr.Number(label="Start Time (in seconds)", value=0)
            video_end_time = gr.Number(label="End Time (in seconds)\n0 for both Start and End Time for extract all", value=0)
            transcribe_video_button = gr.Button("Transcribe from video")
        with gr.Column():
            audio_input = gr.Audio(type="filepath")
            transcribe_audio_button = gr.Button("Transcribe from audio")

    # Display for total runtime and transcript outputs
    transcribe_run_time = gr.Number(label="Total run time : (in seconds)", precision=2)
    with gr.Row():
        with gr.Column():
            text_output = gr.TextArea(label="Transcript", show_copy_button=True, interactive=False)
        with gr.Column():
            text_output_timestamps = gr.TextArea(label="Transcript with timestamps", show_copy_button=True, interactive=False)

    # Linking buttons with transcription functions
    transcribe_video_button.click(transcribe_video, inputs=[video_input, video_start_time, video_end_time], outputs=[text_output, text_output_timestamps, transcribe_run_time])
    transcribe_audio_button.click(transcribe_audio, inputs=audio_input, outputs=[text_output, text_output_timestamps, transcribe_run_time])

if __name__ == "__main__":
    demo.launch()