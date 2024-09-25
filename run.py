import gradio as gr
# import torch
# import torchaudio
import os
import ffmpeg
import time
# from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from faster_whisper import WhisperModel

model = WhisperModel("whisper-th-large-v3-combined-ct2", device="cpu", compute_type="int8")
# model_name = "scb10x/monsoon-whisper-medium-gigaspeech2"
# model_name = "biodatlab/whisper-th-large-v3-combined"
# lang = "th"  # Thai language

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model_name,
#     chunk_length_s=30,
#     device=device,
#     return_timestamps=True,
# )
# pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
#   language=lang,
#   task="transcribe"
# )

def transcribe_audio(audio_filepath):

    start_time = time.time()

    segments, info = model.transcribe(audio_filepath, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    output = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        output.append(segment.text)

    run_time = time.time() - start_time

    return "".join(output), run_time

    # return "Test Transcription from audio"

def transcribe_video(video_filepath):
    start_time = time.time()

    # create tmp directory for intermediate audio file
    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')

    out, _ = (ffmpeg
        .input(video_filepath)
        .output('./tmp/output.wav', ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True)
        )

    output, audio_run_time = transcribe_audio('./tmp/output.wav')

    run_time = time.time() - start_time

    return output, run_time

def copy_from_text_to_text(text):
    return text

def post_process_text(input_text):

    start_time = time.time()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=8000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are text corrector and editor for editing transcription of meeting record.
        Your task are
        - split the text into paragraph
        - add newline between sentences
        - change spell out number into number (0-9)
        by retain the original transcript text as much as possible.
        Retain the language as original one (Thai).
        DO NOT summarize or add any comment.
        Process this text
        
        {text}"""
    )

    llm = ChatOllama(
        # model="gemma2-27b-q8-8k:latest",
        model="gemma2:9b",
        # model="qwen2.5:14b-instruct-q8_0",
        temperature=0,
        # other params...
    )

    chain = prompt | llm | StrOutputParser()

    chunks = text_splitter.split_text(input_text)
    output_text = chain.invoke({"text": "".join(chunks)})
    # output_text = "".join(chain.batch(chunks))

    run_time = time.time() - start_time

    return output_text, run_time

with gr.Blocks() as demo:
    gr.Markdown("# Video/Audio Transcription app")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video()
            transcribe_video_button = gr.Button("Transcribe from video")
        with gr.Column():
            audio_input = gr.Audio(type="filepath")
            transcribe_audio_button = gr.Button("Transcribe from audio")
    text_output = gr.TextArea(label="Transcript", show_copy_button=True, interactive=False)
    transcribe_run_time = gr.Number(label="Total run time : (in seconds)", precision=2)

    gr.Markdown("# Post process")
    copy_from_output_button = gr.Button("Copy from transcript")
    post_process_text_input = gr.TextArea(label="Text input for post-process", show_copy_button=True, info="ถ้าผลลัพธ์เป็นการสรุป ลองตัดข้อความให้สั้นลง")
    post_process_button = gr.Button("Post-process text")
    post_process_text_output = gr.TextArea(label="Text output for post-process")
    post_process_run_time = gr.Number(label="Total run time : (in seconds)", precision=2)

    transcribe_video_button.click(transcribe_video, inputs=video_input, outputs=[text_output, transcribe_run_time])
    transcribe_audio_button.click(transcribe_audio, inputs=audio_input, outputs=[text_output, transcribe_run_time])
    copy_from_output_button.click(copy_from_text_to_text, inputs=text_output, outputs = post_process_text_input)
    post_process_button.click(post_process_text, inputs=post_process_text_input, outputs=[post_process_text_output, post_process_run_time])

if __name__ == "__main__":
    demo.launch()