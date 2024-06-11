from io import BytesIO

import numpy as np
from nlp import Ner
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write

def speech_to_text(recording, recognizer, sample_rate):
    with BytesIO() as buffer_io:
        write(buffer_io, sample_rate, recording)
        with sr.AudioFile(buffer_io) as source:
            audio = recognizer.record(source)
    text = recognizer.recognize_google(audio, language='pt-BR')
    return text

def record_microphone(seconds, sample_rate):
    recording = sd.rec(seconds * sample_rate, samplerate=sample_rate, channels=2, dtype='float64')
    sd.wait()
    recording = (np.iinfo(np.int32).max * (recording / np.abs(recording).max())).astype(np.int32)
    return recording

def find_location(ner, text):
    _, location, _ = ner(text)
    if location != '?':
        return location
    else:
        raise ValueError("Failed to find location")

def find_node(location, node_alias):
    num_nodes = len(node_alias)
    location = location.lower()
    for n, place_alias in enumerate(node_alias):
        for place in place_alias:
            place = place.lower()
            if location in place or place in location:
                return n
    return num_nodes

if __name__ == '__main__':

    print(f"SCRIPT DE ANÁLISE ESTATÍSTICA AO NER")

    possibilities = [
        "",
        "vou para ",
        "estou n",
        "estou perto d",
        "estou ao pé d"
    ]

    node_alias = [
        [
            "a porta"
        ],
        [
            "o meio da sala",
            "o centro da sala",
            "o espaço aberto",
        ],
        [
            "a mesa do Ali",
            "o Baxter",
            "o robô vermelho",
            "a estação de trabalho",
        ],
        [
            "a mesa do Miguel",
            "a bancada dupla",
            "a estante"
        ],
        [
            "a mesa do João",
            "a bancada individual",
            "a mesa redonda",
            "o canto da sala"
        ]
    ]
    sample_rate = 44100
    listening_seconds = 5

    ner = Ner()
    recognizer = sr.Recognizer()

    num_nodes = len(node_alias)
    FAIL_NODE = num_nodes
    count_matrix = np.zeros((num_nodes, num_nodes+1))

    print(f"Quando aparecer no ecrã 'Diga 'algo' ao microfone', tem {listening_seconds} segundos para falar")
    input("Pressione ENTER quando estiver pronto\n")

    for correct_node, node_names in enumerate(node_alias):

        for possible_name in node_names:

            for possible_uterance in possibilities:

                print(f"Diga '{possible_uterance}{possible_name}' ao microfone ({listening_seconds} segundos) e aguarde...", flush=True)

                recording = record_microphone(listening_seconds, sample_rate)
                text = speech_to_text(recording, recognizer, sample_rate)

                ner_location = ner(text)[1]
                if ner_location != '?':
                    understood_node = find_node(ner_location, node_alias)
                    if understood_node != FAIL_NODE:
                        print(f"[DEBUG] NER percebeu {ner_location} -> Nó associado a {ner_location} encontrado ({understood_node}). Registando resultado positivo...")
                    else:
                        print(f"[DEBUG] NER percebeu {ner_location} -> Nó associado a {ner_location} não encontrado. Registando resultado negativo...")
                else:
                    print(f"[DEBUG] NER não identificou o local. Registando resultado negativo...")
                    understood_node = FAIL_NODE

                count_matrix[correct_node, understood_node] += 1
                print("Resultado guardado!")
                input("Pressione ENTER quando estiver pronto para a próxima\n")

    nome = input("Introduz o teu nome (primeiro nome, apelido, tudo junto) (e.g. joaoribeiro) > ")
    np.save(f"count_matrix_{nome}_big", count_matrix)
    print("Já está!")
    print(f"Por favor enviar ficheiro 'count_matrix_{nome}_big.npy' ao João")