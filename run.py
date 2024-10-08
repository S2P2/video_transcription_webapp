import gradio as gr
import os
import ffmpeg
import time
import logging
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

def transcribe_audio(audio_filepath):
    """
    Transcribes the given audio file using the Whisper model.

    Args:
        audio_filepath (str): Path to the audio file to be transcribed.

    Returns:
        tuple: A tuple containing the transcription (str), transcription with timestamps (str),
               and the time taken for transcription (float).
    """
    # Check if the audio file exists
    if not os.path.exists(audio_filepath):
        logging.error(f"Audio file not found: {audio_filepath}")
        return "Error: Audio file not found.", "", 0
    
    try:
        start_time = time.time()
        logging.info(f"Starting audio transcription for: {audio_filepath}")

        # Transcribe the audio using batching for faster process
        segments, info = batched_model.transcribe(audio_filepath, batch_size=16)
        logging.info("Audio transcribe - Done.\nDetected language '%s' with probability %f" % (info.language, info.language_probability))

        output = []  # List to store the transcription text
        output_with_timestamps = []  # List to store the transcription with timestamps
        
        # Iterate through the transcribed segments (generator method for memory efficiency)
        for segment in segments:
            formatted_string = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            logging.debug(formatted_string)
            output.append(segment.text)
            output_with_timestamps.append(formatted_string)

        output = "\n".join(output)
        output_with_timestamps = "\n".join(output_with_timestamps)

        # Ensure the temporary directory exists to save files
        if not os.path.exists('./tmp/'):
            os.makedirs('./tmp/')
            logging.info("Temporary folder created at ./tmp/")
        
        # Save the transcription to a text file
        with open("./tmp/output.txt", "w") as text_file:
            text_file.write(output)
            logging.info("Transcription saved to ./tmp/output.txt")
        
        # Save the transcription with timestamps to a separate text file
        with open("./tmp/output_with_timestamps.txt", "w") as text_file:
            text_file.write(output_with_timestamps)
            logging.info("Transcription with timestamps saved to ./tmp/output_with_timestamps.txt")

        run_time = time.time() - start_time
        logging.info(f"Audio transcription completed in {run_time:.2f} seconds")

        return output, output_with_timestamps, run_time
    
    except Exception as e:
        logging.error(f"Error during audio transcription: {str(e)}")
        return f"Error occured: {str(e)}", "", 0

def transcribe_video(video_filepath, video_start_time, video_end_time):
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
    # Check if the video file exists
    if not os.path.exists(video_filepath):
        logging.error(f"Video file not found: {video_filepath}")
        return "Error: Video file not found.", "", 0
    
    try:
        start_time = time.time()
        logging.info(f"Starting video transcription for: {video_filepath}")

        # Ensure the temporary directory exists to save the extracted audio
        if not os.path.exists('./tmp/'):
            os.makedirs('./tmp/')
            logging.info("Temporary folder created at ./tmp/")

        # Use FFmpeg to extract audio from the full video or from the specified time range
        if video_start_time == 0 and video_end_time == 0:
            logging.info("Extracting audio from the full video...")
            out, _ = (ffmpeg
            .input(video_filepath)
            .output('./tmp/output.wav', ac=1, ar='16k')
            .overwrite_output()
            .run(capture_stdout=True)
            )
        else:
            logging.info(f"Extracting audio from {video_start_time} to {video_end_time} seconds...")
            out, _ = (ffmpeg
            .input(video_filepath, ss=video_start_time, t=video_end_time-video_start_time)
            .output('./tmp/output.wav', ac=1, ar='16k')
            .overwrite_output()
            .run(capture_stdout=True)
            )

        logging.info("Audio extraction from video completed")

        # Transcribe the extracted audio
        output, output_with_timestamps, audio_run_time = transcribe_audio('./tmp/output.wav')

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
            video_start_time = gr.Number(label="Start Time (in seconds)")
            video_end_time = gr.Number(label="End Time (in seconds)\n0 for both Start and End Time for extract all")
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