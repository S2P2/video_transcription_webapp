## Video Transcription Webapp

demo video/audio transcription webapp with gradio using whisper model via faster-whisper, processed video with ffmpeg-python

Using Docker
```
docker run --gpus all --ipc=host -v ./mount:/workspace/project/mount -v ./cache:/root/.cache/huggingface/hub -p 7860:7860 transcription-app
```