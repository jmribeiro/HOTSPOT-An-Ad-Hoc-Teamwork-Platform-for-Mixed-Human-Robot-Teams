from numpy.core._multiarray_umath import ndarray
from tqdm import tqdm

import yaaf.policies
from agents import TEBOPA
from argparse import ArgumentParser
from backend.environments import factory
from backend.environments.predator_prey import teammate_aware_policy, POSSIBLE_TASKS, greedy_policy
from run_hotspot import evaluate_episode
from yaaf import Timestep
from yaaf.agents import Agent
from yaaf.agents.RandomAgent import RandomAgent

import numpy as np
import random


def run_experiment(domain, comms, trials):
    print("Ad Hoc")
    try:
        results = np.load(f"resources/results/{domain}{'-comms' if comms else ''}-{trials}.npy")
    except FileNotFoundError:
        pomdps = [factory[domain](model_id=i, comms=True) for i in range(2)]
        agent = TEBOPA(pomdps)
        possible_tasks = [pomdp for pomdp in agent.pomdps]
        results = np.zeros(trials)
        for trial in tqdm(range(trials)):
            results[trial] = evaluate_episode(agent, random.choice(possible_tasks))
        np.save(f"resources/results/{domain}{'-comms' if comms else ''}-{trials}.npy", results)
    print(results.mean())
    print(results.std())


def partial_baseline(domain, comms, trials):
    print("Known Task")
    try:
        results = np.load(f"resources/results/{domain}-partial-baseline-{trials}.npy")
    except FileNotFoundError:
        pomdps = [factory[domain](model_id=i, comms=True) for i in range(2)]
        results = np.zeros(trials)
        for trial in tqdm(range(trials)):
            pomdp = random.choice(pomdps)
            agent = TEBOPA([pomdp])
            results[trial] = evaluate_episode(agent, pomdp)
        np.save(f"resources/results/{domain}-partial-baseline-{trials}.npy", results)
    print(results.mean())
    print(results.std())


def random_baseline(domain, trials):
    print("Random Baseline")
    try:
        results = np.load(f"resources/results/{domain}-random-{trials}.npy")
    except FileNotFoundError:
        pomdps = [factory[domain](model_id=i, comms=False) for i in range(2)]
        agent = RandomAgent(pomdps[0].num_actions)
        results = np.zeros(trials)
        for trial in tqdm(range(trials)):
            results[trial] = evaluate_episode(agent, random.choice(pomdps))
        np.save(f"resources/results/{domain}-random-{trials}.npy", results)
    print(results.mean())
    print(results.std())


def optimal_baseline(domain, trials):
    print("Optimal Baseline")
    try:
        results = np.load(f"resources/results/{domain}-optimal-{trials}.npy")
    except FileNotFoundError:
        pomdps = [factory[domain](model_id=i, comms=False) for i in range(2)]
        class Optimal(Agent):
            def __init__(self, pomdp):
                super().__init__("Teammate Aware Teammate")
                self.pomdp = pomdp
            def policy(self, observation: ndarray):
                x = self.pomdp.state_index(self.pomdp.state)
                q_values = self.pomdp.q_values[x]
                return yaaf.policies.greedy_policy(q_values)
            def _reinforce(self, timestep: Timestep):
                pass
        results = np.zeros(trials)
        for trial in tqdm(range(trials)):
            pomdp = random.choice(pomdps)
            agent = Optimal(pomdp)
            results[trial] = evaluate_episode(agent, pomdp)
        np.save(f"resources/results/{domain}-optimal-{trials}.npy", results)
    print(results.mean())
    print(results.std())


if __name__ == '__main__':

    opt = ArgumentParser()
    opt = opt.parse_args()
    N = 32

    domain = "pursuit-task"

    print("Setting up algorithm and auxiliary structures")

    run_experiment(domain, True, N);print()
    partial_baseline(domain, True, N);print()
    random_baseline(domain, N);print()
    optimal_baseline(domain, N);print()
