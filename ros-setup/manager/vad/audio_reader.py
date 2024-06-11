# -*- coding: utf-8 -*-
"""
module audio_reader.py
--------------------
Stream audio from a microphone device.
""" 
import numpy as np
import webrtcvad
import pyaudio
import speech_recognition as sr


class AudioReader(object):
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 160
    SLEEP_TIME = 300
    BUFFER = 600

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(3)

        self.stream = None

    def __enter__(self):
        # self.stream = self.audio.open(
        #     format=self.FORMAT, 
        #     channels=self.CHANNELS,
        #     rate=self.RATE,
        #     input=True,
        #     frames_per_buffer=self.CHUNK
        # )
        return self

    def __exit__(self, type, value, traceback):
        # self.stream.stop_stream()
        # self.stream.close()
        # self.audio.terminate()
        pass

    def xread_audio(self, should_filter_activity=True):
        frames = []
        i = 0

        while i < self.BUFFER:
            i += 1
            data = self.stream.read(self.CHUNK)
            
            if should_filter_activity:
                is_speech = self.vad.is_speech(data, self.RATE)
                if is_speech:
                    frames.append(data)
            else:
                frames.append(data)

        if len(frames) > 0:
            audio_data = sr.AudioData(
                np.concatenate(
                    [ np.frombuffer(x, dtype=np.int16) for x in frames ]
                ).tobytes(), 
                sample_rate=self.RATE, 
                sample_width=2
            )
            return audio_data
        else:
            return None

    def read_audio(self, should_filter_activity=True):
        audio_file = sr.AudioFile('vad/data/ao pé da bancada dupla.wav')
        recognizer = sr.Recognizer()
        with audio_file as source:
            audio = recognizer.record(source)
        return audio