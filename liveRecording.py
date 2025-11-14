import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

import time
import wave
from datetime import datetime

import pyaudio

OUTPUT_DIR = r"\\100.103.49.34\test"
CHANNELS = 1
RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 1  # create one file per second


def ensure_output_dir(path=OUTPUT_DIR):
    os.makedirs(path, exist_ok=True)


def record_loop(output_dir=OUTPUT_DIR):
    ensure_output_dir(output_dir)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"Recording live. Files will be saved to '{output_dir}'. Press Ctrl+C to stop.")
    try:
        while True:
            frames = []
            # read RECORD_SECONDS worth of audio in CHUNK-sized reads
            reads = int(RATE / CHUNK * RECORD_SECONDS)
            for _ in range(reads):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.wav"
            filepath = os.path.join(output_dir, filename)

            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames))

            # ensure roughly 1-second cadence; adjust if loop overhead matters
            # sleep a very small time to yield to OS (not required)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    record_loop()