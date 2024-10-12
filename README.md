## Video Transcription Webapp

demo video/audio transcription webapp with gradio and langchain using whisper model via faster-whisper, processed video with ffmpeg-python

install with
```bash
pip install -r requirements.txt
```

to use batched pipeline in faster-whisper
install from main branch
```bash
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```

as to date, the main branch is not published to PyPI yet.
[more detail](https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#install-the-master-branch)

diarization using [pyannote-audio](https://github.com/pyannote/pyannote-audio) and modified code from [speechbox](https://github.com/huggingface/speechbox)