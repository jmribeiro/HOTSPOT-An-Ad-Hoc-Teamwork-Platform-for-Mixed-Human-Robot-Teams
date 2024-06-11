"""
Manager node main
Python 3
"""
import time
import yaml
import numpy as np
from argparse import ArgumentParser

from scipy.io.wavfile import write
import speech_recognition as sr
import sounddevice as sd
from io import BytesIO
import pyttsx3

from nlp import Ner

# ######### #
# Auxiliary #
# ######### #

def say_tts(text):
    print(f"TTS: {text}")
    if tts:
        engine.say(text)
        engine.runAndWait()

  ##########
# Math Utils #
  ##########

def find_next_node(next_index):
    adjacencies = np.where(adjacency_matrix[LAST_KNOWN_ROBOT_LOCATION] == 1)[0]
    downgrade_to_lower_index = int(next_index) >= len(adjacencies)
    next_index = 0 if downgrade_to_lower_index else next_index
    next_node = adjacencies[next_index]
    return next_node

def find_location(location):
    location = location.lower()
    for n, place_alias in enumerate(alias):
        for place in place_alias:
            if location in place or place in location:
                print(f"Found {location} ({place}) (node {n})")
                return n
    print(f"Unknown location '{location}' from {', '.join(places)} and their alias.")
    return -1

  #############
# Manager utils #
  #############


def make_current_observation(dead_reckoning):
    global LAST_SPOKEN_TEXT
    if dead_reckoning is not None and last_spoken_text is None:
        node = -1
    elif dead_reckoning is None and last_spoken_text is not None:
        entity, location, found = ner(last_spoken_text)
        print(f"NER found location {location}")
        node = find_location(location)
        last_spoken_text = None
    else:
        node = LAST_KNOWN_ROBOT_LOCATION
    observation = np.array([node])
    global TIMESTEPS
    timesteps += 1
    return observation


def read_microphone():
    print(f"LISTENING TO MICROPHONE ({listening_seconds} seconds)")
    recording = sd.rec(listening_seconds * sample_rate, samplerate=sample_rate, channels=2, dtype='float64')
    sd.wait()
    print("STOPPED LISTENING TO MICROPHONE")
    recording = (np.iinfo(np.int32).max * (recording / np.abs(recording).max())).astype(np.int32)
    return recording

def speech_to_text(recording):
    with BytesIO() as buffer_io:
        write(buffer_io, sample_rate, recording)
        with sr.AudioFile(buffer_io) as source:
            audio = recognizer.record(source)
    text = recognizer.recognize_google(audio, language='pt-BR')
    return text

def ask_human(prompt):
    try:
        say_tts(prompt)
        recording = read_microphone()
        human_reply = speech_to_text(recording)
        print(f"Human replied with: '{human_reply}'")
        say_tts(f"Disseste que estavas {human_reply}")
        return human_reply
    except Exception as e:
        print(f"Failed to obtain human response ({e})")
        say_tts(f"Não consegui perceber onde estás")
        return None

# ######## #
# Sequence #
# ######## #

def receive_decision_node_message(action):

    global LAST_SPOKEN_TEXT

    print("\n\n\n")
    print("New Timestep")

    print(f"Received action #{action} ({action_meanings[action]}) from Decision node")

    if "move" in action_meanings[action]:
        next_node = find_next_node(action)
        x, y = graph_node_centers_astro_referential[next_node]
        print(f"Awaiting Astro node's message")
        last_spoken_text = None
    elif "locate" in action_meanings[action]:
        last_spoken_text = ask_human("Onde estás?") \
            if action_meanings[action] == "locate human" \
            else ask_human("Onde estou?")
        observation = make_current_observation(dead_reckoning=None)
        print(f"Built observation array {observation}'")
        request_action_from_decision_node(observation)
    else:
        observation = make_current_observation(dead_reckoning=None)
        print(f"Built observation array {observation}'")
        request_action_from_decision_node(observation)

def receive_astro_node_message(message):
    time.sleep(1)
    data = message.data
    dead_reckoning = np.array([float(i) for i in data.split(",")])
    print(f"Received Astro node message: '{dead_reckoning}'")
    observation = make_current_observation(dead_reckoning)
    print(f"Built observation array {observation}'")
    request_action_from_decision_node(observation)

def request_action_from_decision_node(observation: np.ndarray):
    message = " ".join([str(entry) for entry in observation])
    print(f"Sent observation to Decision node: '{message}'")

if __name__ == '__main__':

    opt = ArgumentParser()

    opt.add_argument("-node_name", default="adhoc_pomdp_manager")

    opt.add_argument("-decision_subscriber_topic", default="/adhoc_pomdp/decision_manager")
    opt.add_argument("-decision_publisher_topic", default="/adhoc_pomdp/manager_decision")

    opt.add_argument("-astro_subscriber_topic", default="/adhoc_pomdp/astro_manager")
    opt.add_argument("-astro_publisher_topic", default="/adhoc_pomdp/manager_astro")

    opt.add_argument("-publisher_queue_size", type=int, default=100)
    opt.add_argument("-communication_refresh_rate", type=int, default=10)

    opt = opt.parse_args()

    # ######### #
    # Auxiliary #
    # ######### #

    with open('../resources/config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)["manager"]

    print(f"Initializing auxiliary structures")

    # Text-to-speech
    tts = config["tts"]
    engine = pyttsx3.init()
    engine.setProperty("voice", "portugal")
    engine.setProperty("rate", 150)
    sample_rate = config["mic"]["sample rate"]
    listening_seconds = config["mic"]["listening seconds"]

    LAST_SPOKEN_TEXT = None
    ner = Ner()
    recognizer = sr.Recognizer()

    # Environment Reckon Task #
    action_meanings = (
        "move to lower-index node",
        "move to second-lower-index node",
        "move to third-lower-index node",
        "stay",
        "locate human",
        "locate robot"
    )
    alias = config["node descriptions"]
    places = [place_alias[0] for place_alias in alias]

    adjacency_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ])

    graph_node_centers_astro_referential = np.array(config["graph nodes astro points"])
    seconds_to_move = config["human seconds to move"]

    TIMESTEPS = 0
    LAST_KNOWN_ROBOT_LOCATION = 0

    # ### #
    # ROS #
    # ### #

    print(f"Initializing ROS Node {opt.node_name}")

    # ########### #
    # Subscribers #
    # ########### #

    print(f"Setting up Decision node subscriber (local topic at {opt.decision_subscriber_topic})")
    callback = lambda message: receive_decision_node_message(message)

    print(f"Setting up Astro node subscriber (local topic at {opt.astro_subscriber_topic})")

    # ########## #
    # Publishers #
    # ########## #

    print(f"Setting up Decision node publisher (topic at {opt.decision_publisher_topic})")

    print(f"Setting up Astro node publisher (topic at {opt.astro_publisher_topic})")

    # ### #
    # Run #
    # ### #

    print("Ready")

    while True:
        receive_decision_node_message(action=action_meanings.index("locate human"))
        time.sleep(1.0)