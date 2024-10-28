FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace/project

RUN apt update && apt -y upgrade && apt install -y ffmpeg
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/2dbca5e559a8a36143987e38492cef927a71efd1.tar.gz"
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

RUN mkdir config
RUN mkdir log
COPY run.py .

CMD ["python","run.py"]