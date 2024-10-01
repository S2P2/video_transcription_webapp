import gradio as gr
import os
import ffmpeg
import time
from torch import cuda
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dotenv import load_dotenv

load_dotenv()

WHISPER_CT_MODEL_NAME = "terasut/whisper-th-large-v3-combined-ct2"
COMPUTE_TYPE = "float16"

device = "cuda" if cuda.is_available() else "cpu"
print("Loading "+WHISPER_CT_MODEL_NAME+" process with "+device+", compute_type="+COMPUTE_TYPE)
model = WhisperModel(WHISPER_CT_MODEL_NAME, device=device, compute_type=COMPUTE_TYPE)
batched_model = BatchedInferencePipeline(model=model)
print("Model loaded")

def transcribe_audio(audio_filepath):

    start_time = time.time()

    print("Audio transcribing...")
    segments, info = batched_model.transcribe(audio_filepath, batch_size=16)
    
    print("Audio transcribe - Done.\nDetected language '%s' with probability %f" % (info.language, info.language_probability))

    output = []
    output_with_timestamps = []
    
    for segment in segments:
        formatted_string = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        print(formatted_string)
        output.append(segment.text)
        output_with_timestamps.append(formatted_string)

    output = "\n".join(output)
    output_with_timestamps = "\n".join(output_with_timestamps)


    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')
        print("tmp folder created")
    
    with open("./tmp/output.txt", "w") as text_file:
        text_file.write(output)
        print("transcripts saved to output.txt")
    
    with open("./tmp/output_with_timestamps.txt", "w") as text_file:
        text_file.write(output_with_timestamps)
        print("transcripts saved to output_with_timestamps.txt")

    run_time = time.time() - start_time

    return output, output_with_timestamps, run_time

def transcribe_video(video_filepath, video_start_time, video_end_time):

    start_time = time.time()

    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')
        print("tmp folder created")

    print("Extracting Audio from Video...")
    if video_start_time == 0 and video_end_time == 0:
        print("from full length video...")
        out, _ = (ffmpeg
        .input(video_filepath)
        .output('./tmp/output.wav', ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True)
        )
    else:
        print("from "+str(video_end_time)+" to "+str(video_end_time)+" (s)...")
        out, _ = (ffmpeg
        .input(video_filepath, ss=video_start_time, t=video_end_time-video_start_time)
        .output('./tmp/output.wav', ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True)
        )

    print("Extracting Audio from Video - Done")
    output, output_with_timestamps, audio_run_time = transcribe_audio('./tmp/output.wav')

    run_time = time.time() - start_time

    return output, output_with_timestamps, run_time


with gr.Blocks() as demo:
    gr.Markdown("# Video/Audio Transcription app")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video()
            video_start_time = gr.Number(label="Start Time (in seconds)")
            video_end_time = gr.Number(label="End Time (in seconds)\n0 for both Start and End Time for extract all")
            transcribe_video_button = gr.Button("Transcribe from video")
        with gr.Column():
            audio_input = gr.Audio(type="filepath")
            transcribe_audio_button = gr.Button("Transcribe from audio")
    transcribe_run_time = gr.Number(label="Total run time : (in seconds)", precision=2)
    with gr.Row():
        with gr.Column():
            text_output = gr.TextArea(label="Transcript", show_copy_button=True, interactive=False)
        with gr.Column():
            text_output_timestamps = gr.TextArea(label="Transcript with timestamps", show_copy_button=True, interactive=False)

    transcribe_video_button.click(transcribe_video, inputs=[video_input, video_start_time, video_end_time], outputs=[text_output, text_output_timestamps, transcribe_run_time])
    transcribe_audio_button.click(transcribe_audio, inputs=audio_input, outputs=[text_output, text_output_timestamps, transcribe_run_time])

if __name__ == "__main__":
    demo.launch()