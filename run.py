import gradio as gr
import os
import ffmpeg
import time
import logging
import numpy as np
from typing import Tuple
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pyannote.audio import Pipeline
from torchaudio import functional as F
# from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
logging.basicConfig(filename='transcription_app.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Whisper model configuration
WHISPER_CT_MODEL_NAME = "terasut/whisper-th-large-v3-combined-ct2"
DIARIZER_MODEL_NAME = "pyannote/speaker-diarization-3.1"
COMPUTE_TYPE = "float16"

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
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

# from speechbox.diarize.ASRDiarizationPipeline.preprocess
# (see https://github.com/huggingface/speechbox/blob/e7339dc021c8aa3047f824fb5c24b5b2c8197a76/src/speechbox/diarize.py#L174)
def preprocess_for_diarizer(inputs):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, self.sampling_rate)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets` for better integration
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != self.sampling_rate:
            inputs = F.resample(torch.from_numpy(inputs), in_sampling_rate, self.sampling_rate).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError("We expect a single channel audio input for ASRDiarizePipeline")

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs

# from speechbox.diarize.ASRDiarizationPipeline
# (see https://github.com/huggingface/speechbox/blob/e7339dc021c8aa3047f824fb5c24b5b2c8197a76/src/speechbox/diarize.py#L103)
def diarization(audio_filepath):

    start_time = time.time()

    group_by_speaker = True

    pipeline = Pipeline.from_pretrained(
        DIARIZER_MODEL_NAME,
        use_auth_token=os.getenv("HF_TOKEN")
    )

    pipeline.to(torch.device(device))

    diarization = pipeline(audio_filepath)

    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append({'segment': {'start': segment.start, 'end': segment.end},
                            'track': track,
                            'label': label})

    # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append(
                {
                    "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["start"]},
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append(
        {
            "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["end"]},
            "speaker": prev_segment["label"],
        }
    )

    asr_segments, asr_info = batched_model.transcribe(audio_filepath, batch_size=16)

    transcript = []

    for segment in asr_segments:
        transcript.append({
        "text": segment.text,
        "timestamp": [segment.start, segment.end]
        })

    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
    segmented_preds = []

    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join([chunk["text"] for chunk in transcript[: upto_idx + 1]]),
                    "timestamp": (transcript[0]["timestamp"][0], transcript[upto_idx]["timestamp"][1]),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

    run_time = time.time() - start_time

    return segmented_preds, segmented_preds, run_time

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
            diarizatize_audio_button = gr.Button("Diarize from audio")

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
    diarizatize_audio_button.click(diarization, inputs=audio_input, outputs=[text_output, text_output_timestamps, transcribe_run_time])

if __name__ == "__main__":
    demo.launch()