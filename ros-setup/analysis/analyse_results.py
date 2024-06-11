import scipy.stats

import yaaf
import numpy as np

from decision.balls_pomdp import load_task
from yaaf.agents import GreedyAgent, RandomAgent
from yaaf.visualization import LinePlot
from yaaf.visualization.BarPlot import confidence_bar_plot, bar_plot

NODES = [
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

class Timestep:

    def __init__(self, timestep_str):
        lines = timestep_str.split("\n")
        self.t = int(lines[0])
        self.data = {}
        del lines[0]
        del lines[-1]
        for line in lines:
            key, value = line.split(": ")
            value = value.replace("\n", "")
            value = value.replace("None", "")
            self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return str(self.t) + str(self.data)

class Run:

    def __repr__(self):
        repr = ""
        repr += f"Run {self.directory}\n"
        repr += f"Legal MDP Run: {self.meta['legal']}\n"
        repr += f"Total Timesteps: {self.num_timesteps}\n"
        repr += f"Task: {self.meta['task']}\n"
        return repr

    def __init__(self, directory):
        self.directory = directory
        self.meta = self.parse_meta()
        self.decision_timesteps = self.parse_decision_log()
        self.manager_timesteps = self.parse_manager_log()
        self.human_actions = self.parse_human_actions()
        self.human_utterances, self.can_identify_node = self.parse_human_utterances()
        self.human_nodes = self.parse_human_nodes()
        assert len(self.decision_timesteps) == len(self.manager_timesteps)
        self.num_timesteps = len(self.manager_timesteps)

    @property
    def steps_to_solve(self):
        return len(self.decision_timesteps)

    def parse_human_nodes(self):
        human_nodes = []
        for action in self.human_actions:
            node = action.split(" ")[-1]
            human_nodes.append(int(node))
        return human_nodes

    def parse_human_actions(self):
        actions = []
        with open(f"{self.directory}/human_actions.txt", "r") as file:
            all_lines = file.readlines()
            for line in all_lines:
                timestep, action = line.split(": ")
                action = action.replace("\n", "")
                actions.append(action)
        return actions

    def parse_human_utterances(self):
        utterances = []
        can_identify_nodes = []
        with open(f"{self.directory}/human_utterances.txt", "r") as file:
            all_lines = file.readlines()
            for line in all_lines:
                timestep, utterance = line.split(": ")
                utterance = utterance.replace("\n", "")
                utterance = utterance.replace("\"", "")
                utterance, can_identify_node = utterance.split("; ")
                assert can_identify_node == "yes" or can_identify_node == "no"
                can_identify_node = can_identify_node == "yes"
                utterances.append(utterance)
                can_identify_nodes.append(can_identify_node)
        return utterances, can_identify_nodes

    def parse_decision_log(self):
        timesteps = []
        with open(f"{self.directory}/decision_log.txt", "r") as file:
            all_lines = file.readlines()
            all_lines = "".join(all_lines)
            all_lines = all_lines.replace("t: 0", "0")
            all_lines = all_lines.split("\nt: ")
            for timestep in all_lines:
                timestep = Timestep(timestep)
                timesteps.append(timestep)
        return timesteps

    def parse_manager_log(self):
        timesteps = []
        with open(f"{self.directory}/manager_log.txt", "r") as file:
            all_lines = file.readlines()
            all_lines = "".join(all_lines)
            all_lines = all_lines.replace("t: 0", "0")
            all_lines = all_lines.split("\nt: ")
            for timestep in all_lines:
                timestep = Timestep(timestep)
                timesteps.append(timestep)
        return timesteps

    def parse_meta(self):
        with open(f"{self.directory}/meta.txt", "r") as file:
            meta = {}
            for line in file.readlines():
                key, value = line.split(": ")
                value = value.replace("\n", "")
                meta[key] = value
            return meta

def load_runs(directory):
    runs = []
    for directory in yaaf.subdirectories(directory):
        try:
            run = Run(f"../resources/runs/{directory}")
            runs.append(run)
        except:
            print(f"Unable to load ../resources/runs/{directory}")
            pass
    return runs

def nlp_accuracies(run):

    speech_total = 0
    ner_total = 0
    node_total = 0

    speech_correct = 0
    ner_correct = 0
    node_correct = 0

    for t in range(run.num_timesteps):

        true_spoken_text = run.human_utterances[t]
        true_human_node = run.human_nodes[t]
        human_mentioned_node = run.can_identify_node[t]

        understood_text = run.manager_timesteps[t]['Spoken Text']
        detected_ner_location = run.manager_timesteps[t]['NER Location']
        detected_human_node = int(run.manager_timesteps[t]['Human Node'])
        human_spoke = true_spoken_text != ""

        if human_spoke:

            speech_total += 1

            if true_spoken_text.lower() == understood_text.lower():
                speech_correct += 1

            if human_mentioned_node:
                ner_total += 1
                node_total += 1

                node_alias = NODES[true_human_node]
                for alias in node_alias:
                    if detected_ner_location in alias:
                        ner_correct += 1

                if true_human_node == detected_human_node:
                    node_correct += 1

    return (speech_correct, speech_total), (ner_correct, ner_total), (node_correct, node_total)

def task_identification_accuracy(run):

    true_task = int(run.meta["task"])
    total = run.num_timesteps
    correct = 0
    entropy_run = np.zeros(run.num_timesteps + 1)
    entropy_run[0] = 1.0
    steps_to_identify = run.num_timesteps
    for t in range(run.num_timesteps):
        beliefs = run.decision_timesteps[t]["Beliefs"]
        beliefs = beliefs.split(" (")[0]
        beliefs = beliefs.replace("[", "")
        beliefs = beliefs.replace("]", "")
        beliefs = np.fromstring(beliefs, sep=",")
        mlt = beliefs.argmax()
        entropy = scipy.stats.entropy(beliefs, base=beliefs.size)
        entropy_run[t + 1] = entropy
        if mlt == true_task:
            correct += 1
            steps_to_identify = min(steps_to_identify, t)
        else:
            steps_to_identify = run.num_timesteps
    last = mlt == true_task

    return (correct, total), last, steps_to_identify, entropy_run

def plot_entropy(runs):

    max_timesteps = 0
    entropy_runs = []

    for run in runs:
        max_timesteps = max(max_timesteps, run.num_timesteps)
        (_speech_correct, _speech_total), (_ner_correct, _ner_total), (_node_correct, _node_total) = nlp_accuracies(run)
        (_correct_timesteps,_total_timesteps), identified, steps_to_identify, entropy_run = task_identification_accuracy(run)
        entropy_runs.append(entropy_run)

    max_timesteps += 1
    plot = LinePlot("Average TEBOPA Entropy over Possible Tasks", "Timestep", "Entropy", max_timesteps, ymin=-0.01, ymax=1.2)
    for entropy_run in entropy_runs:
        if entropy_run.size < max_timesteps:
            padded_entropy_run = np.zeros(max_timesteps)
            padded_entropy_run[0:entropy_run.size] = entropy_run
            padded_entropy_run[entropy_run.size:] = entropy_run[-1]
            entropy_run = padded_entropy_run
        print(entropy_run[-3:])
        plot.add_run("", entropy_run, color="red")
    plot.show()
    yaaf.mkdir("../resources/plots")
    filename = "../resources/plots/Entropy.pdf"
    plot.savefig(filename)

def plot_steps(runs):
    results = get_steps(runs)
    N = len(runs)
    names = [
        "TEBOPA\nSteps To\nSolve Task\n(real world)",
        "TEBOPA\nSteps To\nIdentify Correct Task\n(real world)",
        "Optimal Policy\nSteps To\nSolve Task\n(simulated)",
        "Random Policy\nSteps To\nSolve Task\n(simulated)"
    ]
    means = [result.mean() for result in results]
    yaaf.mkdir("../resources/plots")
    filename = "../resources/plots/Steps.pdf"
    confidence_bar_plot(names, means, N, f"Agent Comparison (N={N})", None, "Steps", True, filename, colors=["r", "magenta", "g", "b"])

def plot_accuracies(runs):

    speech_correct = speech_total = 0
    ner_correct = ner_total = 0
    node_correct = node_total = 0
    for run in runs:
        (_speech_correct, _speech_total), (_ner_correct, _ner_total), (_node_correct, _node_total) = nlp_accuracies(run)
        speech_correct += _speech_correct
        speech_total += _speech_total
        ner_correct += _ner_correct
        ner_total += _ner_total
        node_correct += _node_correct
        node_total += _node_total
    speech_accuracy = speech_correct / speech_total
    ner_accuracy = ner_correct / ner_total
    nodes_accuracy = node_correct / node_total
    names = ["Speech Recognition", "Named Entity Recognition", "Node Identification"]
    results = [speech_accuracy, ner_accuracy, nodes_accuracy]
    print(names)
    print(results)
    yaaf.mkdir("../resources/plots")
    filename = "../resources/plots/Accuracies.pdf"
    bar_plot(names, results, f"Accuracies", "Module", "Accuracy", True, filename, colors=["r", "g", "b"])

def simulate_agent(task, initial_state, agent_name):
    env = load_task(task, "../resources/cache")
    initial_belief = np.zeros(env.num_states)
    x0 = env.state_index(initial_state)
    initial_belief[x0] = 1.0
    env.reset(initial_belief)
    agent = GreedyAgent(env) if agent_name == "greedy" else RandomAgent(env.num_actions)
    state = initial_state
    steps = 0
    terminal = False
    while not terminal:
        action = agent.action(state)
        next_obs, reward, _, info = env.step(action)
        next_state = env.state
        steps += 1
        state = next_state
        terminal = (state[2:] == 2).all()
    return steps

def get_steps(runs):
    greedy_steps = []
    random_steps = []
    tebopa_steps = []
    tebopa_steps_identify = []
    for run in runs:
        initial_state = run.meta["initial state"]
        initial_state = initial_state.replace("[", "")
        initial_state = initial_state.replace("]", "")
        initial_state = np.fromstring(initial_state, dtype=int, sep=',')
        task = int(run.meta["task"])
        tebopa_steps.append(run.steps_to_solve)
        greedy_steps.append(simulate_agent(task, initial_state, "greedy"))
        random_steps.append(simulate_agent(task, initial_state, "random"))
        _, identified, steps_to_identify, _ = task_identification_accuracy(run)
        tebopa_steps_identify.append(steps_to_identify)
    return np.array(tebopa_steps), np.array(tebopa_steps_identify), np.array(greedy_steps), np.array(random_steps)

if __name__ == '__main__':
    runs = load_runs("../resources/runs")
    plot_accuracies(runs)
    plot_entropy(runs)
    plot_steps(runs)
