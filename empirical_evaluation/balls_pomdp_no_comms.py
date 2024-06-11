import random
from itertools import product
from typing import Sequence

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy

import yaaf
from agents import QMDPAgent, TEQMDP, TEBOPA
from pomdp import PartiallyObservableMarkovDecisionProcess
from yaaf.agents import GreedyAgent
from yaaf.environments.markov import MarkovDecisionProcess

BASE_ID = "balls"
BALL_NODES = (
    [0, 1, 2],
    [1, 2, 3]#,
    #[1, 2, 4]
)
GROUND, HELD, DISPOSED = 0, 1, 2
GROUND_COST, HAND_COST = -1, -3
ADJACENCY_MATRIX = np.array(
    [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ]
)

ACTION_MEANINGS = [
    "move to lower-index node",
    "move to second-lower-index node",
    "move to third-lower-index node",
    "stay",
]

# ### #
# MDP #
# ### #

def generate_states(ball_nodes):

    def valid_state(state):

        p2, ball_status = state[1], state[2:]

        if (ball_status == HELD).sum() > 1:
            # Human can't hold two balls
            return False
        elif (ball_status == HELD).sum() == 1:
            # Human holding ball requires p2 == ball node
            ball_index = tuple(ball_status).index(HELD)
            ball_node = ball_nodes[ball_index]
            return ball_node == p2
        else:
            return True

    num_nodes = ADJACENCY_MATRIX.shape[0]
    ball_statuses = 3   # Ground, Held, Disposed
    states = [
        np.array([p1, p2, b1, b2, b3])
        for p1 in range(num_nodes)
        for p2 in range(num_nodes)
        for b1 in range(ball_statuses)
        for b2 in range(ball_statuses)
        for b3 in range(ball_statuses)
    ]

    return [state for state in states if valid_state(state)]

def generate_transitions(states, ball_nodes):

    def human_next_node(state):
        if ball_nodes == [0, 1, 2]:
            return task_0_next_human_node(state)
        elif ball_nodes == [1, 2, 3]:
            return task_1_next_human_node(state)
        elif ball_nodes == [1, 2, 4]:
            return task_2_next_human_node(state)
        else:
            raise ValueError("Not implemented")

    def robot_next_node(current_node, action):
        adjacencies = np.where(ADJACENCY_MATRIX[current_node] == 1)[0]
        downgrade_to_lower_index = int(action) >= len(adjacencies)
        action = 0 if downgrade_to_lower_index else action
        return adjacencies[action]

    def ball_transitions(state, ball_index, a_meaning):

        p1, p2, _, _, _ = state
        ball_status = state[2+ball_index]
        ball_location = ball_nodes[ball_index]
        ground = ball_status == GROUND
        held = ball_status == HELD
        disposed = ball_status == DISPOSED
        human_on_ball = ball_location == p2
        robot_on_ball = ball_location == p1
        robot_stays = "stay" in a_meaning

        if disposed:                                                          # Already disposed
            next_ball_state = DISPOSED
        elif ground and not human_on_ball:                                    # Human not on ball
            next_ball_state = GROUND
        elif ground and human_on_ball:                                        # Human can pickup ball
            next_ball_state = HELD
        elif held and human_on_ball and not robot_on_ball:                    # Human holding ball
            next_ball_state = HELD
        elif held and human_on_ball and robot_on_ball and not robot_stays:    # Human holding ball
            next_ball_state = HELD
        elif held and robot_on_ball and robot_stays:                          # Human disposing ball
            next_ball_state = DISPOSED
        else:
            print("Ball location", ball_location)
            print("Ground?", ground)
            print("Held?", held)
            print("Disposed?", disposed)
            print("Human on ball?", human_on_ball)
            print("Robot on ball?", robot_on_ball)
            print("Robot stays?", robot_stays)
            raise ValueError("Debug this!")

        return next_ball_state

    def possible_transitions(state, a_meaning):

        p1, p2, b1, b2, b3 = state

        if "move" in a_meaning:
            next_p1 = robot_next_node(p1, ACTION_MEANINGS.index(a_meaning))
        else:
            next_p1 = p1

        next_p2 = human_next_node(state)
        if next_p2 != p2:
            possible_next_p2 = {
                next_p2: 0.95,
                p2: 0.05
            }
        else:
            possible_next_p2 = {p2: 1.0}

        next_b1 = ball_transitions(state, 0, a_meaning)
        next_b2 = ball_transitions(state, 1, a_meaning)
        next_b3 = ball_transitions(state, 2, a_meaning)

        transitions = {}
        for next_p2, prob in possible_next_p2.items():
            next_state = np.array([next_p1, next_p2, next_b1, next_b2, next_b3])
            y = yaaf.ndarray_index_from(states, next_state)
            transitions[y] = prob

        return transitions

    num_states = len(states)
    num_actions = len(ACTION_MEANINGS)
    P = np.zeros((num_actions, num_states, num_states))
    for a, a_meaning in enumerate(ACTION_MEANINGS):
        for x, state in enumerate(states):
            for y, probability in possible_transitions(state, a_meaning).items():
                P[a, x, y] = probability
            if not np.isclose(P[a, x].sum(), 1.0):
                print(P[a, x].sum())
                raise ValueError("Failed transitions")
    return P

def generate_rewards(states):
    num_states = len(states)
    num_actions = len(ACTION_MEANINGS)
    R = np.zeros((num_states, num_actions))
    for x, state in enumerate(states):
        ball_status = state[2:]
        all_disposed = (ball_status == DISPOSED).all()
        if all_disposed:
            R[x, :] = 0.0
        else:
            balls_on_ground = (ball_status == GROUND).sum()
            balls_on_hand = (ball_status == HELD).sum()
            R[x, :] = (balls_on_ground * GROUND_COST) + (balls_on_hand * HAND_COST)

    R_norm = np.zeros((num_states, num_actions))
    min = R.min()
    max = R.max()
    div = max - min
    for x in range(num_states):
        R_norm[x, :] = (R[x, :] - min) / div
    return R_norm

def generate_miu(states):
    num_states = len(states)
    miu = np.zeros(num_states)
    valid_initial_states = []
    for x, state in enumerate(states):
        ball_status = state[2:]
        if (ball_status == GROUND).all():
            valid_initial_states.append(x)
    p = 1 / len(valid_initial_states)
    for x in valid_initial_states:
        miu[x] = p
    return miu

def generate_mdp(ball_nodes):
    if ball_nodes not in BALL_NODES:
        raise ValueError("Invalid task")
    states = generate_states(ball_nodes)
    P = generate_transitions(states, ball_nodes)
    R = generate_rewards(states)
    miu = generate_miu(states)
    id = f"balls-mdp-{''.join([str(node) for node in ball_nodes])}-v1"
    mdp = MarkovDecisionProcess(id, states, tuple(range(len(ACTION_MEANINGS))), P, R, 0.95, miu, action_meanings=ACTION_MEANINGS)
    return mdp

# ##### #
# POMDP #
# ##### #

def generate_observations():
    num_nodes = ADJACENCY_MATRIX.shape[0]
    observations = [
        np.array([op1, op2])
        for op1 in range(-1, num_nodes)
        for op2 in range(-1, num_nodes)
    ]
    return observations

def generate_observation_probabilities(states, observations):


    def possible_observations(next_state, a_meaning):

        p1, p2, b1, b2, b3 = next_state

        op1_obs = {p1: 0.99, -1: 0.01}

        if p1 == p2:
            op2_obs = {p2: 1.0}
        else:
            op2_obs = {-1: 1.0}

        obs_prob = {}
        for op1, op1_prob in op1_obs.items():
            for op2, op2_prob in op2_obs.items():
                observation = np.array([op1, op2])
                z = yaaf.ndarray_index_from(observations, observation)
                obs_prob[z] = op1_prob * op2_prob

        return obs_prob

    num_actions = len(ACTION_MEANINGS)
    num_states = len(states)
    num_observations = len(observations)
    O = np.zeros((num_actions, num_states, num_observations))
    for a, a_meaning in enumerate(ACTION_MEANINGS):
        for y, next_state in enumerate(states):
            for z, probability in possible_observations(next_state, a_meaning).items():
                O[a, y, z] = probability
            if not np.isclose(O[a, y].sum(), 1.0):
                print(O[a, y].sum())
                raise ValueError("Failed observations")
    return O

def generate_pomdp(ball_nodes):
    if ball_nodes not in BALL_NODES:
        raise ValueError("Invalid task")
    states = generate_states(ball_nodes)
    observations = generate_observations()
    P = generate_transitions(states, ball_nodes)
    O = generate_observation_probabilities(states, observations)
    R = generate_rewards(states)
    miu = generate_miu(states)
    id = f"balls-pomdp-no-comms-{''.join([str(node) for node in ball_nodes])}-v1"
    pomdp = PartiallyObservableMarkovDecisionProcess(id, states, tuple(range(len(ACTION_MEANINGS))), observations, P, O, R, 0.95, miu, action_meanings=ACTION_MEANINGS)
    return pomdp

# ######### #
# Auxiliary #
# ######### #

def task_0_next_human_node(state):

    # 0, 1, 2

    _, p2, b1, b2, b3 = state

    all_disposed = b1 == DISPOSED and b2 == DISPOSED and b3 == DISPOSED

    if all_disposed:
        return p2
    else:

        # Door
        if p2 == 0:
            if b1 != DISPOSED:
                return p2
            else:
                return 1

        # Middle
        elif p2 == 1:
            if b2 != DISPOSED:
                return p2
            elif b1 != DISPOSED:
                return 0
            else:
                return 2

        # Baxter
        elif p2 == 2:
            if b3 != DISPOSED:
                return p2
            else:
                return 1

        elif p2 == 3:
            return 1

        elif p2 == 4:
            return 3

        else:
            raise ValueError("Unexpected")

def task_1_next_human_node(state):

    # 1, 2, 3

    _, p2, b1, b2, b3 = state

    all_disposed = b1 == DISPOSED and b2 == DISPOSED and b3 == DISPOSED

    if all_disposed:
        return p2
    else:

        # Middle
        if p2 == 1:
            if b1 != DISPOSED:
                return p2
            elif b2 != DISPOSED:
                return 2
            else:
                return 3

        # Baxter
        elif p2 == 2:
            if b2 != DISPOSED:
                return p2
            else:
                return 1

        # Miguel
        elif p2 == 3:
            if b3 != DISPOSED:
                return p2
            else:
                return 1

        elif p2 == 0:
            return 1

        elif p2 == 4:
            return 3

        else:
            raise ValueError("Unexpected")

def task_2_next_human_node(state):

    # 1, 2, 4

    _, p2, b1, b2, b3 = state

    all_disposed = b1 == DISPOSED and b2 == DISPOSED and b3 == DISPOSED

    if all_disposed:
        return p2
    else:

        # Middle
        if p2 == 1:
            if b1 != DISPOSED:
                return p2
            elif b2 != DISPOSED:
                return 2
            else:
                return 3

        # Baxter
        elif p2 == 2:
            if b2 != DISPOSED:
                return p2
            else:
                return 1

        # Joao
        elif p2 == 4:
            if b3 != DISPOSED:
                return p2
            else:
                return 3

        elif p2 == 0:
            return 1

        elif p2 == 3:
            if b1 != DISPOSED or b2 != DISPOSED:
                return 1
            else:
                return 4

        else:
            raise ValueError("Unexpected")

def draw_state(state, ball_locations, node_meanings=None, title=None):

    if (state == -1).all():
        return

    interest_color = ("red", "pink", "lightgreen")
    interest_tags = ("on floor", "on hand", "disposed")

    graph = nx.DiGraph()
    labels = {}
    x_robot, x_human, waste_status = state[0], state[1], state[2:]
    num_nodes = ADJACENCY_MATRIX.shape[0]
    node_meanings = node_meanings or [str(n) for n in range(num_nodes)]
    colors = []
    for n in range(num_nodes):
        graph.add_node(n)
        label = node_meanings[n].replace(' ', '\n')
        labels[n] = f"[{n}: {label}]"
        if x_robot == n:
            labels[n] += f"\nR"
        if x_human == n:
            label = "\nH" if 'R' not in labels[n] else ', H'
            labels[n] += f"{label}"

        if n in ball_locations:
            if waste_status[ball_locations.index(n)] == 0:
                colors.append(interest_color[0])
                labels[n] = labels[n].replace("]", f"]\n({interest_tags[0]})")
            elif waste_status[ball_locations.index(n)] == 1:
                colors.append(interest_color[1])
                labels[n] = labels[n].replace("]", f"]\n({interest_tags[1]})")
            else:
                colors.append(interest_color[2])
                labels[n] = labels[n].replace("]", f"]\n({interest_tags[2]})")
        else:
            colors.append("white")

    rows, cols = np.where(ADJACENCY_MATRIX == 1)
    graph.add_edges_from(zip(rows.tolist(), cols.tolist()))
    plt.figure()
    node_draw_pos = nx.spring_layout(graph)
    fig = nx.draw_networkx_nodes(graph, node_color=colors, pos=node_draw_pos, node_size=8000)
    fig.set_edgecolor('k')
    plt.gcf().set_size_inches(10, 10, forward=True)
    nx.draw_networkx_edges(graph, node_draw_pos, width=1.0, node_size=8000, arrowsize=10)
    nx.draw_networkx_labels(graph, node_draw_pos, labels, font_size=12)
    plt.axis('off')
    plt.title(title or f"State: {state}")
    plt.show()
    plt.close()

# ##### #
# Tasks #
# ##### #

def load_task(task_index, cache_directory):
    ball_nodes = BALL_NODES[task_index]
    id = f"balls-pomdp-no-comms-{''.join([str(node) for node in ball_nodes])}-v1"
    directory = f"{cache_directory}/pomdps/{BASE_ID}"
    try:
        pomdp = PartiallyObservableMarkovDecisionProcess.load(f"{directory}/{id}")
    except FileNotFoundError:
        pomdp = generate_pomdp(ball_nodes)
        pomdp.save(directory)
    pomdp.spec.goal = ball_nodes
    return pomdp

# #### #
# Main #
# #### #

def render_mdp(ball_nodes):
    mdp = generate_mdp(ball_nodes)
    draw = lambda state: draw_state(state, ball_nodes, node_meanings=["door", "middle", "baxter", "miguel", "joao"])
    state = mdp.reset()
    draw(state)
    agent = GreedyAgent(mdp)
    input()
    terminal = False
    while not terminal:
        action = agent.action(state)
        next_state, reward, _, _ = mdp.step(action)
        state = next_state
        terminal = (state[2:] == DISPOSED).all()
        draw(next_state)
        input()

def render_pomdp(ball_nodes, heuristic=TEQMDP):

    pomdp = generate_pomdp(ball_nodes)
    draw = lambda state: draw_state(state, ball_nodes, node_meanings=["door", "middle", "baxter", "miguel", "joao"])
    pomdp.reset()
    agent = heuristic(pomdp)

    draw(pomdp.state)
    print(entropy(agent.belief, base=pomdp.num_states))

    input()
    terminal = False
    while not terminal:
        action = agent.action(None)
        next_obs, reward, _, info = pomdp.step(action)
        terminal = (pomdp.state[2:] == DISPOSED).all()
        timestep = yaaf.Timestep(None, action, reward, next_obs, terminal, info)
        agent.reinforcement(timestep)
        draw(pomdp.state)
        print()
        print("Actio:", pomdp.action_meanings[action])
        print("Obser:", next_obs)
        print("State:", pomdp.state)
        print("MLSTT:", pomdp.states[agent.belief.argmax()], f"({1-entropy(agent.belief, base=pomdp.num_states)}% certainty)")
        input()

def render_pomdp_adhoc(ball_nodes):

    pomdp = generate_pomdp(ball_nodes)
    draw = lambda state: draw_state(state, ball_nodes, node_meanings=["door", "middle", "baxter", "miguel", "joao"])
    pomdp.reset()

    agent = TEBOPA([generate_pomdp(goal) for goal in BALL_NODES], [len(ACTION_MEANINGS)-1])

    draw(pomdp.state)

    input()
    terminal = False
    while not terminal:
        action = agent.action(None)
        next_obs, reward, _, info = pomdp.step(action)
        terminal = (pomdp.state[2:] == DISPOSED).all()
        timestep = yaaf.Timestep(None, action, reward, next_obs, terminal, info)
        agent.reinforcement(timestep)
        draw(pomdp.state)
        mlt = agent.pomdp_probabilities.argmax()
        print()
        print(agent.pomdp_probabilities)
        print("Actio:", pomdp.action_meanings[action])
        print("Obser:", next_obs)
        print("State:", pomdp.state)
        print("MLSTT:", agent.pomdps[mlt].states[agent.beliefs[mlt].argmax()])
        input()

if __name__ == '__main__':

    ball_nodes = [0, 1, 2]#, 3]
    #render_mdp(ball_nodes)
    #render_pomdp(ball_nodes)
    render_pomdp_adhoc(ball_nodes)
    print("Success")

