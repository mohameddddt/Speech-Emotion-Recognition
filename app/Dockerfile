# Hugging Face Space (sdk: docker). Built from app/build/space/ by app/deploy.py.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \
    HOST=0.0.0.0 PORT=7860 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# Download the encoder weights into the image so a cold start doesn't fetch them.
RUN python server.py --download

EXPOSE 7860
CMD ["python", "server.py"]
