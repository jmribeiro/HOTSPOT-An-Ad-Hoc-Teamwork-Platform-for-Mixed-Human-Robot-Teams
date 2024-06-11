"""
Manager node main
Python 3
"""
import time
import yaml
import numpy as np
from argparse import ArgumentParser

import rospy
from std_msgs.msg import String

from scipy.io.wavfile import write
import speech_recognition as sr
import sounddevice as sd
from io import BytesIO
import pyttsx3

#NER-FIXME
from nlp import Ner

# ######### #
# Auxiliary #
# ######### #

def say_tts(text):
    rospy.loginfo(f"TTS: {text}")
    if tts:
        engine.say(text)
        engine.runAndWait()

  ##########
# Math Utils #
  ##########

def row_index(row, matrix):
    if not isinstance(row, np.ndarray): row = np.array(row)
    possible = np.where(np.all(matrix == row, axis=1))
    if len(possible) != 1: raise ValueError("Not a valid row in the matrix")
    else: return possible[0]

def find_next_node(next_index):
    adjacencies = np.where(adjacency_matrix[LAST_KNOWN_ROBOT_LOCATION] == 1)[0]
    downgrade_to_lower_index = int(next_index) >= len(adjacencies)
    next_index = 0 if downgrade_to_lower_index else next_index
    next_node = adjacencies[next_index]
    return next_node

def closest_node(point, centers):
    node = None
    smallest = np.inf
    for n, center in enumerate(centers):
        distance = np.linalg.norm((center-point), 2)
        if distance < smallest:
            smallest = distance
            node = n
    return node

def find_location(location):
    location = location.lower()
    for n, place_alias in enumerate(NODE_ALIAS):
        for place in place_alias:
            place = place.lower()
            if location in place or place in location:
                rospy.loginfo(f"Found {location} ({place}) (node {n})")
                return n
    rospy.loginfo(f"Unknown location '{location}' from {', '.join(NODE_NAMES)} and their alias.")
    raise ValueError("Unknown location")

  #############
# Manager utils #
  #############

def read_astro_node(dead_reckoning_coordinates):
    global LAST_KNOWN_ROBOT_LOCATION
    try:
        n_astro = closest_node(dead_reckoning_coordinates, GRAPH_NODES_CENTER_TO_ASTRO_POINTS)
        rospy.loginfo(f"Astro is closest to {NODE_NAMES[n_astro]}")
        #say_tts(f"O Astro está {NODE_NAMES[n_astro]}")
        LAST_KNOWN_ROBOT_LOCATION = n_astro
    except ValueError:
        n_astro = LAST_KNOWN_ROBOT_LOCATION
        rospy.logwarn(f"Astro's coordinates {dead_reckoning_coordinates} do not map to any valid known node. "
                      f"Using last known location ({NODE_NAMES[n_astro]})")
    return n_astro

def make_current_observation(dead_reckoning_coordinates, last_spoken_text):

    if dead_reckoning_coordinates is not None:
        robot_node = read_astro_node(dead_reckoning_coordinates)
    else:
        robot_node = LAST_KNOWN_ROBOT_LOCATION

    if last_spoken_text is not None:
        entity, location, found = ner(last_spoken_text)
        rospy.loginfo(f"NER found location {location}")
        try:
            human_node = find_location(location)
            say_tts(f"Disseste {NODE_NAMES[human_node]}")
        except ValueError:
            human_node = -1
            say_tts(f"Não consegui perceber o que disseste")
    else:
        location = None
        human_node = -1

    global LAST_RFID, BALLS_CAUGHT, JUST_CAUGHT_BALL
    JUST_CAUGHT_BALL = False
    if LAST_RFID is not None:
        rfid_sensor = LAST_RFID
        BALLS_CAUGHT += 1
        say_tts("Bola Recebida!")
        JUST_CAUGHT_BALL = True
    else:
        rfid_sensor = 0
    observation = (robot_node, human_node, rfid_sensor)

    LAST_RFID = None
    global TIMESTEPS

    print(flush=True)
    print(f"t: {TIMESTEPS}", flush=True)
    print(f"Dead Reckoning: {dead_reckoning_coordinates}", flush=True)
    print(f"Robot Node: {robot_node}", flush=True)
    print(f"Spoken Text: {last_spoken_text}", flush=True)
    print(f"NER Location: {location}", flush=True)
    print(f"Human Node: {human_node}", flush=True)
    print(f"RFID Sensor: {bool(rfid_sensor)}", flush=True)
    print(f"Observation: {observation}", flush=True)

    TIMESTEPS += 1
    return observation

def read_microphone():
    rospy.loginfo(f"LISTENING TO MICROPHONE ({listening_seconds} seconds)")
    recording = sd.rec(listening_seconds * sample_rate, samplerate=sample_rate, channels=2, dtype='float64')
    sd.wait()
    rospy.loginfo("STOPPED LISTENING TO MICROPHONE")
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
        rospy.loginfo(f"Human replied with: '{human_reply}'")
        return human_reply
    except Exception as e:
        rospy.loginfo(f"Failed to obtain human response ({e})")
        return None

def ask_human_node(human_speech_to_text):
    entity, location, found = ner(human_speech_to_text)
    rospy.loginfo(f"NER found location {location}")
    try:
        node = find_location(location)
        say_tts(f"Disseste {NODE_NAMES[node]}")
    except ValueError:
        node = -1
        say_tts(f"Não consegui perceber o que disseste")
    return node

# ######## #
# Sequence #
# ######## #

def request_action_from_decision_node(observation: tuple):
    countdown(5)
    message = str(observation)
    rospy.loginfo(f"Sent observation to Decision node: '{message}'")
    decision_publisher.publish(message)

def countdown(seconds):
    for s in range(seconds):
        time.sleep(1)
        #print(seconds - s, end=" ")

def receive_decision_node_message(action: String):

    global LAST_ACTION

    last_spoken_text = None
    dead_reckoning_coordinates = None

    rospy.loginfo("\n\n\n")
    rospy.loginfo("New Timestep")

    action = int(action.data)

    if action == 42:
        say_tts("Apanhamos as três bolas! Obrigado!")
        exit(0)

    rospy.loginfo(f"Received action #{action} ({ACTION_MEANINGS[action]}) from Decision node")

    if "move" in ACTION_MEANINGS[action]:
        next_node = find_next_node(action)
        x, y = GRAPH_NODES_CENTER_TO_ASTRO_POINTS[next_node]
        send_astro_robot_move_order(f"go to {x}, {y}")
        dead_reckoning_coordinates = np.array([x, y])
        say_tts("Podes fazer a tua ação.")
        say_tts(f"Indo {NODE_ALIAS[next_node][0]}")
        countdown(10)
    elif "locate" in ACTION_MEANINGS[action]:
        last_spoken_text = ask_human("O que vais fazer agora?")
        say_tts("Podes fazer a tua ação.")
        countdown(6)
    else:
        say_tts(f"Vou ficar quieto em {NODE_ALIAS[LAST_KNOWN_ROBOT_LOCATION][0]}")
        say_tts("Podes fazer a tua ação.")
        countdown(6)

    observation = make_current_observation(dead_reckoning_coordinates, last_spoken_text)
    rospy.loginfo(f"Built observation array {observation}'")
    if BALLS_CAUGHT == 3:
        say_tts("Apanhamos as três bolas! Obrigado!")
        exit(0)
    LAST_ACTION = ACTION_MEANINGS[action]

    request_action_from_decision_node(observation)

def send_astro_robot_move_order(order: str):
    rospy.loginfo(f"Sent Astro node order {order}")
    astro_publisher.publish(order)

def receive_rfid_notification(message: str):
    global LAST_RFID
    LAST_RFID = 1


if __name__ == '__main__':

    opt = ArgumentParser()

    opt.add_argument("-node_name", default="adhoc_pomdp_manager")

    opt.add_argument("-decision_subscriber_topic", default="/adhoc_pomdp/decision_manager")
    opt.add_argument("-decision_publisher_topic", default="/adhoc_pomdp/manager_decision")

    opt.add_argument("-astro_publisher_topic", default="/adhoc_pomdp/manager_astro")
    opt.add_argument("-astro_rfid_topic", default="/idmind_rfid")

    opt.add_argument("-publisher_queue_size", type=int, default=100)
    opt.add_argument("-communication_refresh_rate", type=int, default=10)

    opt = opt.parse_args()
    print("Teste0", flush=True)

    rospy.init_node(opt.node_name)

    # ######### #
    # Auxiliary #
    # ######### #

    with open('../resources/config.yml', 'r') as file: config = yaml.load(file, Loader=yaml.FullLoader)["manager"]
    print("Teste1", flush=True)
    rospy.loginfo(f"Initializing auxiliary structures")
    print("Teste2", flush=True)
    # Text-to-speech
    tts = config["tts"]
    engine = pyttsx3.init()
    engine.setProperty("voice", "portugal")
    engine.setProperty("rate", 150)
    sample_rate = config["mic"]["sample rate"]
    listening_seconds = config["mic"]["listening seconds"]

    ner = Ner()
    recognizer = sr.Recognizer()

    ACTION_MEANINGS = (
        "move to lower-index node",
        "move to second-lower-index node",
        "move to third-lower-index node",
        "stay",
        "locate human",
    )
    NODE_ALIAS = config["node descriptions"]
    NODE_NAMES = [place_alias[0] for place_alias in NODE_ALIAS]

    adjacency_matrix = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ])

    GRAPH_NODES_CENTER_TO_ASTRO_POINTS = np.array(config["graph nodes astro points"])
    seconds_to_move = config["seconds to move"]

    TIMESTEPS = 0
    BALLS_CAUGHT = 0
    JUST_CAUGHT_BALL = False
    LAST_ACTION = None
    LAST_RFID = None
    JUST_ACCEPTED = False

    LAST_KNOWN_ROBOT_LOCATION = -1
    while LAST_KNOWN_ROBOT_LOCATION == -1:
        text = ask_human("Onde estou?")
        if text is not None:
            LAST_KNOWN_ROBOT_LOCATION = ask_human_node(text)

    # ### #
    # ROS #
    # ### #

    rospy.loginfo(f"Initializing ROS Node {opt.node_name}")

    # ########### #
    # Subscribers #
    # ########### #

    rospy.loginfo(f"Setting up Decision node subscriber (local topic at {opt.decision_subscriber_topic})")
    decision_subscriber = rospy.Subscriber(opt.decision_subscriber_topic, String, receive_decision_node_message)

    # ########## #
    # Publishers #
    # ########## #

    rospy.loginfo(f"Setting up Decision node publisher (topic at {opt.decision_publisher_topic})")
    decision_publisher = rospy.Publisher(opt.decision_publisher_topic, String, queue_size=opt.publisher_queue_size)

    rospy.loginfo(f"Setting up Astro node publisher (topic at {opt.astro_publisher_topic})")
    astro_publisher = rospy.Publisher(opt.astro_publisher_topic, String, queue_size=opt.publisher_queue_size)

    rospy.loginfo(f"Setting up Astro node RFID subscriber (local topic at {opt.astro_rfid_topic})")
    rfid_subscriber = rospy.Subscriber(opt.astro_rfid_topic, String, receive_rfid_notification)

    # ### #
    # Run #
    # ### #

    #say_tts("Podes-te mexer, força!")
    countdown(5)

    rospy.loginfo("Ready")
    rate = rospy.Rate(opt.communication_refresh_rate)
    rospy.spin()
