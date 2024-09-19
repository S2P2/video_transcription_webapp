import gradio as gr
import torch
import torchaudio
import os
import ffmpeg
from transformers import pipeline

# model_name = "scb10x/monsoon-whisper-medium-gigaspeech2"
model_name = "biodatlab/whisper-th-large-v3-combined"
lang = "th"  # Thai language

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_name,
    chunk_length_s=30,
    device=device,
    return_timestamps=True,
)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
  language=lang,
  task="transcribe"
)

def transcribe_speech(filepath):
    # Load the .wav file using torchaudio
    waveform, sample_rate = torchaudio.load(filepath)

    # If the audio is stereo, convert it to mono by averaging the channels
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Ensure the waveform is in the expected format
    audio = {"array": waveform.squeeze().numpy(), "sampling_rate": sample_rate}

    output = pipe(
        audio,
        batch_size=8,
    )
    return output["text"]

def extract_and_transcribe(video_filepath, start_time, end_time):

    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')

    if start_time == 0 and end_time == 0:
        out, _ = (ffmpeg
        .input(video_filepath)
        .output('./tmp/output.wav', ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True)
        )
    else:
        out, _ = (ffmpeg
        .input(video_filepath, ss=start_time, t=end_time-start_time)
        .output('./tmp/output.wav', ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True)
        )

    output = transcribe_speech('./tmp/output.wav')
    
    return output

# Build Gradio interface
interface = gr.Interface(
    extract_and_transcribe, 
    [
        gr.Video(label="Upload Video (.mp4)"), 
        gr.Number(label="Start Time (in seconds)"), 
        gr.Number(label="End Time (in seconds)\n0 for both Start and End Time for extract all")
    ], 
    "text"
)

if __name__ == "__main__":
    # Launch the interface
    interface.launch()
