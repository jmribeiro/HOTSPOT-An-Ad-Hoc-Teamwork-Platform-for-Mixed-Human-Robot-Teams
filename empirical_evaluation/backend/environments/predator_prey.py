import itertools

import numpy as np

import yaaf
from backend.environments.loader import try_load

from backend.environments.pursuit_py.astar import A_star_search
from backend.environments.pursuit_py.utils import agent_directions, action_meanings
from backend.pomdp import PartiallyObservableMarkovDecisionProcess
from yaaf.policies import random_policy, policy_from_action

NAME = "predator-prey"

ABSORBENT = [42, 42, 42, 42]
WIN_REWARD = 1.0
POSSIBLE_TASKS = {
    # Maps task to relative to prey state
    "n/s": [(0, -1, 0, 1), (0, 1, 0, -1)],
    "w/e": [(-1, 0, 1, 0), (1, 0, -1, 0)],
    "sw/ne": [(1, -1, -1, 1), (-1, 1, 1, -1)],
    "nw/se": [(-1, -1, 1, 1), (1, 1, -1, -1)]
}

POSSIBLE_TEAMMATES = ["greedy", "teammate aware"]
POSSIBLE_MODELS = list(itertools.product(POSSIBLE_TEAMMATES, list(POSSIBLE_TASKS.keys())))
ACTION_MEANINGS = [
    "up",
    "down",
    "left",
    "right",
    "stay"
]


def create(model_id, comms, world_size):

    assert comms in [True, False]
    possible_models = [0, 1, 2, 3]
    if model_id > len(possible_models):
        raise ValueError(f"Task id for pursuit must be between 0 and {len(POSSIBLE_MODELS)-1} (got {model_id})")

    noise = 30
    id = f"{NAME}{'-comms' if comms else ''}-{world_size[0]}-{noise}-v{model_id}"
    teammate, task = POSSIBLE_MODELS[possible_models[model_id]]
    pomdp = try_load(lambda: PursuitPOMDP(id, task, teammate, comms, world_size), id)
    return pomdp


class PursuitPOMDP(PartiallyObservableMarkovDecisionProcess):

    def __init__(self,
                 id: str,
                 task,
                 teammate,
                 comms: bool,
                 world_size: tuple,
                 discount_factor: float = 0.95):
        self.task = task
        states, action_meanings, transition_probabilities, rewards_matrix, miu = generate_mdp(task, teammate, world_size)
        self._world_size = world_size

        if comms:
            # Locate = Stay
            A, X, X = transition_probabilities.shape

            pomdp_transition_probabilities = np.zeros((A+1, X, X))
            pomdp_rewards_matrix = np.zeros((X, A+1))

            for a in range(A):
                pomdp_transition_probabilities[a] = transition_probabilities[a]
                pomdp_rewards_matrix[:, a] = rewards_matrix[:, a]

            pomdp_transition_probabilities[A] = transition_probabilities[A-1]
            pomdp_rewards_matrix[:, A] = rewards_matrix[:, A-1]

        else:
            pomdp_transition_probabilities = transition_probabilities
            pomdp_rewards_matrix = rewards_matrix

        observations = self._generate_observations()
        pomdp_action_meanings = action_meanings if not comms else action_meanings + ["locate"]
        observation_probabilities = self._generate_observation_probabilities(states, pomdp_action_meanings, observations)

        super().__init__(id,
                         states, tuple(range(len(pomdp_action_meanings))), observations,
                         pomdp_transition_probabilities, observation_probabilities,
                         pomdp_rewards_matrix, discount_factor, miu,
                         action_meanings=pomdp_action_meanings)

        model_id = int(id.split("-v")[1])
        self.teammate, self.task = POSSIBLE_MODELS[model_id]
        self.world_size = world_size

    def _generate_observations(self):
        observations = [
            np.array([dx1, dy1, dxp, dyp])

            for dx1 in list(range(-2, 3)) + [42]
            for dy1 in list(range(-2, 3)) + [42]

            for dxp in list(range(-2, 3)) + [42]
            for dyp in list(range(-2, 3)) + [42]
        ]

        return observations

    def  _generate_observation_probabilities(self, states, action_meanings, observations):

        num_actions = len(action_meanings)
        num_states = len(states)
        num_observations = len(observations)

        def is_in_surrounding(a, b):
            x1, y1 = a
            x2, y2 = b
            # Define the range of the surrounding cells
            row_range = range(y1 - 1, y1 + 2)
            col_range = range(x1 - 1, x1 + 2)
            # Check if the target cell is within the surrounding cells
            return y2 in row_range and x2 in col_range

        def possible_observations(action, next_state):

            if tuple(next_state) == tuple(ABSORBENT):
                return {(42, 42, 42, 42): 1.0}

            else:
                dx1, dy1, dxp, dyp = next_state

                agent = (0, 0)
                teammate = dx1, dy1
                prey = dxp, dyp

                if action == "locate":
                    teammate_possibilities = {(dx1, dy1): 1.0}
                    prey_possibilities = {(dxp, dyp): 1.0}
                else:
                    teammate_possibilities = {
                        (dx1, dy1): 0.7,
                        (42, 42): 0.30
                    } if is_in_surrounding(agent, teammate) else {(42, 42): 1.0}

                    prey_possibilities = {
                        (dxp, dyp): 0.70,
                        (42, 42): 0.30
                    } if is_in_surrounding(agent, prey) else {(42, 42): 1.0}

                possible_combinations = {}
                for teammate_obs, teammate_obs_prob in teammate_possibilities.items():
                    for prey_obs, prey_obs_prob in prey_possibilities.items():
                        dx1, dy1 = teammate_obs
                        dxp, dyp = prey_obs
                        next_obs = (dx1, dy1, dxp, dyp)
                        prob = teammate_obs_prob * prey_obs_prob
                        if next_obs not in possible_combinations: possible_combinations[next_obs] = prob
                        else: possible_combinations[next_obs] += prob
                return possible_combinations

        O = np.zeros((num_actions, num_states, num_observations))
        for a in range(num_actions):
            for y in range(num_states):
                next_state = states[y]
                action = action_meanings[a]
                for observation, observation_probability in possible_observations(action, next_state).items():
                    z = yaaf.ndarray_index_from(observations, np.array(observation))
                    O[a, y, z] = observation_probability
                assert np.isclose(O[a, y].sum(), 1.0)
        return O

    def render(self, mode="human"):

        state = self.state
        num_rows, num_columns = self._world_size
        dx1, dy1, dxp, dyp = state
        cxa, cya, cxb, cyb = capture_points(state, self.task, self._world_size)

        def draw_cell(cell):
            x, y = cell
            x -= int(num_columns / 2)
            y -= int(num_rows / 2)
            if 0 == x and 0 == y: return 'A'
            elif dx1 == x and dy1 == y: return 'T'
            elif dxp == x and dyp == y: return "P"
            elif cxa == x and cya == y: return self.task.split("/")[0]
            elif cxb == x and cyb == y: return self.task.split("/")[1]
            else: return " "

        def draw_row_border(columns):
            border = ""
            for y in range(columns): border += "+---"
            border += "+\n"
            return border

        display = ""
        display += f" {' '.join([f' {i} ' for i in range(num_columns)])}\n"
        for row in range(num_rows):
            display += draw_row_border(num_columns)
            display += "|"
            for col in range(num_columns):
                cell = col, row
                cell_display = draw_cell(cell)
                display += " " if col == 0 else "+ "
                display += f"{cell_display} "
            display += f"| {row}\n"
        display += draw_row_border(num_columns)
        print(display)


def generate_mdp(task, teammate, world_size):
    states = generate_state_space(world_size)
    action_meanings = ACTION_MEANINGS

    megacache_directory = f"resources/cache/shared/predator_prey_{task.replace('/', '')}_{world_size[0]}x{world_size[1]}"
    yaaf.mkdir(megacache_directory)

    try:
        transition_probabilities = np.load(f"{megacache_directory}/P.npy")
    except FileNotFoundError:
        transition_probabilities = generate_transition_probabilities_cube(world_size, states, action_meanings, task, teammate)
        np.save(f"{megacache_directory}/P.npy", transition_probabilities)

    try:
        rewards_matrix = np.load(f"{megacache_directory}/R.npy")
    except FileNotFoundError:
        rewards_matrix = generate_rewards_matrix(states, world_size, action_meanings, task)
        np.save(f"{megacache_directory}/R.npy", rewards_matrix)

    miu = generate_miu(states, world_size, task)
    return states, action_meanings, transition_probabilities, rewards_matrix, miu

def generate_state_space(world_size):

    num_columns, num_rows = world_size

    full_states = [
        np.array((x_agent, y_agent, x_teammate, y_teammate, x_prey, y_prey))
        for x_agent in range(num_columns) for y_agent in range(num_rows)
        for x_teammate in range(num_columns) for y_teammate in range(num_rows)
        for x_prey in range(num_columns) for y_prey in range(num_rows)
    ]

    relative_states = []
    for full_state in full_states:
        relative_state = relative_distance_feature_extractor(full_state, world_size)
        relative_state = tuple(relative_state)
        if relative_state not in relative_states:
            relative_states.append(relative_state)

    states = [np.array(ABSORBENT)]
    for relative_state in relative_states:
        state = np.array(relative_state)
        states.append(state)

    #print(f"Reduced {len(full_states)+1} into {len(states)}")

    return states

def generate_transition_probabilities_cube(world_size, states, action_meanings, task, teammate):

    num_states = len(states)
    num_actions = len(action_meanings)
    num_columns, num_rows = world_size

    def compute_next_state(state, action_meaning_0, action_meaning_1, action_meaning_p):

        is_goal_state = prey_captured(state, world_size, task)
        is_reset_state = tuple(state) == tuple(ABSORBENT)

        # Absorbent Transition
        if is_goal_state or is_reset_state:
            return np.array(ABSORBENT)

        # Regular Transition
        else:

            action_deltas = {
                "up": (0, -1),
                "down": (0, 1),
                "left": (-1, 0),
                "right": (1, 0),
                "stay": (0, 0),
            }

            dx1, dy1, dxp, dyp = state
            delta_x0, delta_y0 = action_deltas[action_meaning_0]
            delta_x1, delta_y1 = action_deltas[action_meaning_1]
            delta_xp, delta_yp = action_deltas[action_meaning_p]

            dx1_next = dx1 + (delta_x1 - delta_x0)
            dy1_next = dy1 + (delta_y1 - delta_y0)

            dxp_next = dxp + (delta_xp - delta_x0)
            dyp_next = dyp + (delta_yp - delta_y0)

            # Toroidal check
            dx1_next = toroidal_check(dx1_next, num_rows)
            dy1_next = toroidal_check(dy1_next, num_columns)
            dxp_next = toroidal_check(dxp_next, num_rows)
            dyp_next = toroidal_check(dyp_next, num_columns)

            next_state = dx1_next, dy1_next, dxp_next, dyp_next

            return np.array(next_state)

    P_mmdp = np.zeros((num_actions, num_actions, num_actions, num_states, num_states))
    for a0 in range(num_actions):
        for a1 in range(num_actions):
            for ap in range(num_actions):
                for x in range(num_states):
                    state = states[x]
                    action_meaning_0 = action_meanings[a0]
                    action_meaning_1 = action_meanings[a1]
                    action_meaning_p = action_meanings[ap]
                    next_state = compute_next_state(state, action_meaning_0, action_meaning_1, action_meaning_p)
                    y = yaaf.ndarray_index_from(states, next_state)
                    P_mmdp[a0, a1, ap, x, y] = 1.0

    teammate_policy = {
        "greedy": greedy_policy,
        "teammate aware": teammate_aware_policy,
        "random": lambda state, world_size, task: random_policy(num_actions)
    }
    prey_policy = random_policy(num_actions)

    P = np.zeros((num_actions, num_states, num_states))
    for a0 in range(num_actions):
        for x in range(num_states):
            state = states[x]
            pi_1 = teammate_policy[teammate](state, world_size, task)
            for y in range(num_states):
                for a1 in range(num_actions):
                    for ap in range(num_actions):
                        P[a0, x, y] += P_mmdp[a0, a1, ap, x, y] * pi_1[a1] * prey_policy[ap]
    P = np.round(P, 3)
    return P

def relative_distance_prey(state, world_size):
    dx1, dy1, dxp, dyp = state
    dxp1 = dx1 - dxp
    dyp1 = dy1 - dyp
    dxp1 = toroidal_check(dxp1, world_size[1])
    dyp1 = toroidal_check(dyp1, world_size[0])
    dxp0 = -1 * dxp
    dyp0 = -1 * dyp
    state = dxp0, dyp0, dxp1, dyp1
    return state

def relative_distance_teammate(state, world_size):
    dx1, dy1, dxp, dyp = state
    dx1p = dxp - dx1
    dy1p = dyp - dy1
    dx1p = toroidal_check(dx1p, world_size[1])
    dy1p = toroidal_check(dy1p, world_size[0])
    dx10 = -1 * dx1
    dy10 = -1 * dy1
    state = dx10, dy10, dx1p, dy1p
    return state

def prey_captured(state, world_size, task):
    state_relative_prey = relative_distance_prey(state, world_size)
    return state_relative_prey in POSSIBLE_TASKS[task]

def generate_rewards_matrix(states, world_size, action_meanings, task):

    num_actions = len(action_meanings)
    num_states = len(states)

    # Any movement costs -1
    R = np.zeros((num_states, num_actions))

    # Actions on prey captured yield 100
    for x, state in enumerate(states):
        if prey_captured(state, world_size, task):
            R[x, :] = WIN_REWARD

    # Actions on reset yield 0
    y_reset = yaaf.ndarray_index_from(states, np.array(ABSORBENT))
    R[y_reset, :] = 0.0

    return R

def generate_miu(states, world_size, task):
    starting_states = []
    for x, state in enumerate(states):
        ended = prey_captured(state, world_size, task) or tuple(state) == tuple(ABSORBENT)
        if not ended:
            starting_states.append(x)
    miu = np.zeros(len(states))
    for x in starting_states:
        miu[x] = 1 / len(starting_states)
    return miu

def toroidal_distance_1d(x, y, size):
    return toroidal_check(y - x, size)

def toroidal_distance_2d(source, target, world_size):
    num_columns, num_rows = world_size
    x0, y0 = source
    x1, y1 = target
    delta_x = toroidal_distance_1d(x0, x1, num_columns)
    delta_y = toroidal_distance_1d(y0, y1, num_rows)
    return delta_x, delta_y

def relative_distance_feature_extractor(state, world_size):
    x0, y0, x1, y1, xp, yp = state
    agent = x0, y0
    teammate = x1, y1
    prey = xp, yp
    delta_teammate = toroidal_distance_2d(agent, teammate, world_size)
    delta_prey = toroidal_distance_2d(agent, prey, world_size)
    state = np.array((delta_teammate + delta_prey))
    return state

def toroidal_check(d, size):
    if abs(d) == size:
        return 0
    if d < 0 and (size % 2 == 0) and abs(d) == (size / 2):
        return int(size / 2)
    elif abs(d) > int(size/2):
        if d < 0:
            return (-1 * d) - int(size/2)
        else:
            return int(size / 2) - d
    else:
        return d

def capture_points(state, task, world_size):

    _, _, dxp, dyp = state

    capture_offsets = {
        "n": (0, -1),
        "s": (0, 1),
        "w": (-1, 0),
        "e": (1, 0),
        "nw": (-1, -1),
        "ne": (1, -1),
        "se": (1, 1),
        "sw": (-1, 1)
    }
    point_a_str, point_b_str = task.split("/")
    point_a_offset = capture_offsets[point_a_str]
    point_b_offset = capture_offsets[point_b_str]

    point_a_x = toroidal_check(dxp + point_a_offset[0], world_size[0])
    point_a_y = toroidal_check(dyp + point_a_offset[1], world_size[1])

    point_b_x = toroidal_check(dxp + point_b_offset[0], world_size[0])
    point_b_y = toroidal_check(dyp + point_b_offset[1], world_size[1])

    return point_a_x, point_a_y, point_b_x, point_b_y

# TEAMMATES #

def closest_capture_location(relative_state, task, world_size):
    from scipy.spatial.distance import cityblock
    source = (0, 0)
    cxa, cya, cxb, cyb = capture_points(relative_state, task, world_size)
    capture_a, capture_b = (cxa, cya), (cxb, cyb)
    distance_a, distance_b = cityblock(source, capture_a), cityblock(source, capture_b)
    return capture_a if distance_a <= distance_b else capture_b
    
def greedy_policy(state, world_size, task):

    num_actions = len(ACTION_MEANINGS)

    state_relative_teammate = relative_distance_teammate(state, world_size)
    capture_cell = closest_capture_location(state_relative_teammate, task, world_size)

    def greedy_direction(source, target):
        num_columns, num_rows = world_size[1], world_size[0]
        dx_forward = max(0, source[0] - target[0])
        dx_backward = min(target[0] - source[0], num_columns - 1)
        dy_forward = max(0, source[1] - target[1])
        dy_backward = min(target[1] - source[1], num_rows - 1)
        if dx_forward < dx_backward: return "right"
        elif dx_backward < dx_forward: return "left"
        elif dy_forward < dy_backward: return "down"
        elif dy_backward < dy_forward: return "up"
        else: return "stay"

    direction = greedy_direction((0, 0), capture_cell)
    policy = np.zeros(num_actions)
    policy[ACTION_MEANINGS.index(direction)] = 1.0
    return policy

def teammate_aware_policy(state, world_size, task):

    num_actions = len(ACTION_MEANINGS)
    state_relative_teammate = relative_distance_teammate(state, world_size)
    dx0, dy0, dxp, dyp = state_relative_teammate
    teammate = (0, 0)
    agent = dx0, dy0

    capture_cell = closest_capture_location(state_relative_teammate, task, world_size)

    action, _ = A_star_search(teammate, {agent}, capture_cell, world_size)

    if action is None:
        return random_policy(num_actions)
    elif action == (0, 0):
        stay = ACTION_MEANINGS.index("stay")
        return policy_from_action(stay, num_actions)
    else:
        a_backend = agent_directions().index(action)
        a_backend_meaning = action_meanings()[a_backend]
        a = ["Up", "Down", "Left", "Right"].index(a_backend_meaning)
        return policy_from_action(a, num_actions)
