from scipy.io.wavfile import write
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import yaml
from io import BytesIO

with open('resources/config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)
seconds = config["manager"]["mic"]["listening seconds"]
sample_rate = config["manager"]["mic"]["sample rate"]
print(f"Listening for {seconds} seconds")
recording = sd.rec(seconds * sample_rate, samplerate=sample_rate, channels=2, dtype='float64')
sd.wait()
print("Stopped listening")
recording = (np.iinfo(np.int32).max * (recording / np.abs(recording).max())).astype(np.int32)
with BytesIO() as buffer:
    write(buffer, sample_rate, recording)
    recognizer = sr.Recognizer()
    with sr.AudioFile(buffer) as source: audio = recognizer.record(source)
    text = recognizer.recognize_google(audio, language='pt-BR')
    print(f"Audio converted to text: {text}")
