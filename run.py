import gradio as gr
import os
import ffmpeg
import time
import logging
import tomlkit
from typing import List, Tuple
from torch import cuda
from faster_whisper import WhisperModel, BatchedInferencePipeline
import json # Added
from omegaconf import OmegaConf # Added
from nemo.collections.asr.models.msdd_models import NeuralDiarizer # Added

# Set up logging configuration
logging.basicConfig(filename='transcription_app_gr5.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

try:
    config = tomlkit.loads(open("config.toml").read())
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
    '''
    Converts time in seconds to a more readable format (H:M:S) using the time module.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Time in H:M:S format.
    '''
    try:
        time_struct = time.gmtime(int(seconds))
        return time.strftime("%H:%M:%S", time_struct)
    except Exception as e:
        logging.error(f"Error formatting time: {str(e)}")
        raise

def format_transcription(segments: List, offset: float = 0.0) -> Tuple[str, str]:
    '''
    Formats the transcription segments into plain text and timestamped formats, with an optional offset applied to timestamps.

    Args:
        segments (list): List of transcribed segments.
        offset (float, optional): Time in seconds to shift the timestamps to account for video trimming. Defaults to 0.

    Returns:
        tuple: (Plain transcription, transcription with timestamps).
    '''
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

        return "\\n".join(output), "\\n".join(output_with_timestamps)
    except Exception as e:
        logging.error(f"Error formatting transcription: {str(e)}")
        raise

def diarize_audio(audio_filepath: str, output_dir: str, config_path: str) -> str:
    '''
    Performs speaker diarization on an audio file using NVIDIA NeMo.

    Args:
        audio_filepath: Path to the input audio file.
        output_dir: Directory to store temporary files and prediction RTTM.
        config_path: Path to the NeMo configuration YAML file (e.g., 'diar_infer_telephonic.yaml').

    Returns:
        Path to the predicted RTTM file.
    '''
    try:
        logging.info(f"Starting audio diarization for: {audio_filepath}")
        os.makedirs(output_dir, exist_ok=True)

        # Create a temporary manifest file for NeMo
        meta = {
            'audio_filepath': os.path.abspath(audio_filepath),
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': None,  # Set to None for NeMo to estimate
            'rttm_filepath': None, # No ground truth RTTM for inference
            'uem_filepath': None   # No UEM file for inference
        }
        manifest_filepath = os.path.join(output_dir, "input_manifest.json")
        with open(manifest_filepath, 'w') as fp:
            json.dump(meta, fp)
            fp.write('\n')

        # Load and update NeMo configuration
        cfg = OmegaConf.load(config_path)
        
        cfg.diarizer.manifest_filepath = manifest_filepath
        cfg.diarizer.out_dir = output_dir
        
        cfg.diarizer.speaker_embeddings.model_path = 'titanet_large'
        cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet'
        cfg.diarizer.msdd_model.model_path = 'diar_msdd_telephonic'

        cfg.diarizer.oracle_vad = False
        cfg.diarizer.clustering.parameters.oracle_num_speakers = False
        
        cfg.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7] 
        cfg.num_workers = 1 # Safer default for now

        logging.info(f"NeMo Diarizer Config for {audio_filepath}: {OmegaConf.to_yaml(cfg.diarizer)}")

        diarizer_model = NeuralDiarizer(cfg=cfg)

        logging.info(f"Starting NeMo diarization process for {audio_filepath}...")
        diarizer_model.diarize()
        logging.info(f"NeMo diarization process completed for {audio_filepath}.")

        audio_file_basename = os.path.splitext(os.path.basename(audio_filepath))[0]
        predicted_rttm_filepath = os.path.join(output_dir, "pred_rttms", f"{audio_file_basename}.rttm")

        if not os.path.exists(predicted_rttm_filepath):
            logging.error(f"Predicted RTTM file NOT FOUND at: {predicted_rttm_filepath}")
            # Attempt to find any .rttm file in the directory as a fallback for debugging
            pred_rttms_dir = os.path.join(output_dir, 'pred_rttms')
            if os.path.exists(pred_rttms_dir):
                logging.warning(f"Files found in {pred_rttms_dir}: {os.listdir(pred_rttms_dir)}")
                # Example: if manifest name was used by NeMo, like 'input_manifest.rttm'
                fallback_rttm = os.path.join(pred_rttms_dir, "input_manifest.rttm")
                if os.path.exists(fallback_rttm):
                    logging.warning(f"Fallback RTTM found: {fallback_rttm}. Using this one.")
                    predicted_rttm_filepath = fallback_rttm
                else:
                     raise FileNotFoundError(f"Predicted RTTM file {predicted_rttm_filepath} not found, and fallback input_manifest.rttm also not found.")
            else:
                 raise FileNotFoundError(f"Predicted RTTM file {predicted_rttm_filepath} not found and pred_rttms directory {pred_rttms_dir} does not exist.")


        logging.info(f"Predicted RTTM file for {audio_filepath} generated at: {predicted_rttm_filepath}")
        return predicted_rttm_filepath

    except Exception as e:
        logging.error(f"Error during audio diarization for {audio_filepath}: {str(e)}", exc_info=True)
        # Clean up manifest if it exists
        if 'manifest_filepath' in locals() and os.path.exists(manifest_filepath):
            try:
                os.remove(manifest_filepath)
                logging.info(f"Cleaned up manifest file: {manifest_filepath}")
            except Exception as e_clean:
                logging.warning(f"Could not clean up manifest file {manifest_filepath}: {e_clean}")
        raise

def transcribe(audio_filepath: str) -> Tuple[str, str, float]:
     >> new_run.py
echo '    Transcribes audio to text using the Faster Whisper model.' >> new_run.py
echo '' >> new_run.py
echo '    Args:' >> new_run.py
echo '        audio_filepath (str): Path to the audio file.' >> new_run.py
echo '' >> new_run.py
echo '    Returns:' >> new_run.py
echo '        tuple: (Plain transcription, transcription with timestamps, run time).' >> new_run.py
echo  
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
     >> new_run.py
echo '    Transcribes video to text by first extracting audio and then using the Faster Whisper model.' >> new_run.py
echo '' >> new_run.py
echo '    Args:' >> new_run.py
echo '        video_filepath (str): Path to the video file.' >> new_run.py
echo '        video_start_time (float): Start time for the video clip to transcribe.' >> new_run.py
echo '        video_end_time (float): End time for the video clip to transcribe.' >> new_run.py
echo '' >> new_run.py
echo '    Returns:' >> new_run.py
echo '        tuple: (Plain transcription, transcription with timestamps, run time).' >> new_run.py
echo  
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
        gr.TextArea(label="Transcript", show_copy_button=True), 
        gr.TextArea(label="Transcript with timestamps", show_copy_button=True), 
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
        gr.TextArea(label="Transcript", show_copy_button=True), 
        gr.TextArea(label="Transcript with timestamps", show_copy_button=True), 
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
