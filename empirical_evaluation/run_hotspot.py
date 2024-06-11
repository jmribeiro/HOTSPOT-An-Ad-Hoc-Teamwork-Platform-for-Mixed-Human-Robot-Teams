import math
from argparse import ArgumentParser

from tqdm import tqdm

from yaaf.agents.RandomAgent import RandomAgent
from balls_pomdp import BALL_NODES, load_task as load_comms
from balls_pomdp_no_comms import load_task as load_no_comms

from agents import TEBOPA
from yaaf import Timestep
import numpy as np
import random
import yaml


def setup_adhoc_agent(no_comms=False):
    cache_directory = "resources/cache"
    num_tasks = len(BALL_NODES)
    pomdps = []
    drawers = []
    for t in range(num_tasks):
        pomdp = load_no_comms(t, cache_directory) if no_comms else load_comms(t, cache_directory)
        pomdps.append(pomdp)
    agent = TEBOPA(pomdps, [] if no_comms else [pomdps[0].action_meanings.index("locate human")])
    agent.drawers = drawers
    try:
        from yaaf import rmdir
        rmdir("Estados mais prováveis")
    except:
        pass
    return agent


def evaluate_episode(agent, env, max=100):
    if hasattr(agent, "reset"):
        agent.reset()
    env.reset()
    steps = 0
    terminal = False
    while not terminal:
        action = agent.action(None)
        #print(f"t={steps}: {env.action_meanings[action]}")
        next_obs, reward, _, info = env.step(action)
        #print(f"t={steps}: {next_obs}")
        timestep = Timestep(None, action, reward, next_obs, terminal, info)
        agent.reinforcement(timestep)
        terminal = reward == 1.0
        steps += 1
        if steps == max:
            break
    return steps


if __name__ == '__main__':

    opt = ArgumentParser()
    opt = opt.parse_args()
    N = 32

    print("With communication")
    try:
        results = np.load("resources/results/hotspot-32trials-comms.npy")
    except FileNotFoundError:
        agent = setup_adhoc_agent()
        possible_tasks = [pomdp for pomdp in agent.pomdps]
        action_meanings = possible_tasks[-1].action_meanings
        results = np.zeros(N)
        for trial in range(N):
            results[trial] = evaluate_episode(agent, random.choice(possible_tasks))
        np.save("resources/results/hotspot-32trials-comms.npy", results)
    print(results)
    print(results.mean())
    print(results.std())

    print("Without communication")
    try:
        results = np.load("resources/results/hotspot-32trials-no-comms.npy")
    except FileNotFoundError:
        agent = setup_adhoc_agent(no_comms=True)
        possible_tasks = [pomdp for pomdp in agent.pomdps]
        results = np.zeros(N)
        for trial in tqdm(range(N)):
            results[trial] = evaluate_episode(agent, random.choice(possible_tasks))
        np.save("resources/results/hotspot-32trials-no-comms.npy", results)
    print(results)
    print(results.mean())
    print(results.std())

    print("Non-Ad Hoc, With communication")
    try:
        results = np.load("resources/results/hotspot-32trials-comms-no-adhoc.npy")
    except FileNotFoundError:
        agent = setup_adhoc_agent()
        possible_tasks = [pomdp for pomdp in agent.pomdps]
        action_meanings = possible_tasks[-1].action_meanings
        results = np.zeros(N)
        for trial in range(N):
            pomdp = random.choice(possible_tasks)
            agent = TEBOPA([pomdp], [pomdp.action_meanings.index("locate human")])
            results[trial] = evaluate_episode(agent, pomdp)
        np.save("resources/results/hotspot-32trials-comms-no-adhoc.npy", results)
    print(results)
    print(results.mean())
    print(results.std())

    print("Non-Ad Hoc, Without communication")
    try:
        results = np.load("resources/results/hotspot-32trials-no-comms-no-adhoc.npy")
    except FileNotFoundError:
        agent = setup_adhoc_agent(no_comms=True)
        possible_tasks = [pomdp for pomdp in agent.pomdps]
        results = np.zeros(N)
        for trial in tqdm(range(N)):
            pomdp = random.choice(possible_tasks)
            agent = TEBOPA([pomdp], [])
            results[trial] = evaluate_episode(agent, pomdp)
        np.save("resources/results/hotspot-32trials-no-comms-no-adhoc.npy", results)
    print(results)
    print(results.mean())
    print(results.std())

    print("Random")
    try:
        results = np.load("resources/results/hotspot-32trials-random.npy")
    except FileNotFoundError:
        agent = RandomAgent(len(possible_tasks[0].action_meanings))
        results = np.zeros(N)
        for trial in tqdm(range(N)):
            results[trial] = evaluate_episode(agent, random.choice(possible_tasks))
        np.save("resources/results/hotspot-32trials-random.npy", results)
    print(results)
    print(results.mean())
    print(results.std())
