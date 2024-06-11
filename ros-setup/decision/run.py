"""
Decision node main
Python 3
"""
import time
from ast import literal_eval as make_tuple

import rospy
import yaml
from scipy.stats import entropy
from std_msgs.msg import String
from argparse import ArgumentParser
import numpy as np
from agents import TEBOPA
from balls_pomdp import BALL_NODES
from yaaf import Timestep


def send_manager_message(message: String):
    manager_publisher.publish(message)


def receive_manager_message(message: String):
    global LAST_ACTION, T
    #rospy.loginfo("\n\n\n")
    #rospy.loginfo("New Timestep")
    next_obs = make_tuple(message.data)
    next_obs = np.array(next_obs)[:-1]
    agent.reinforcement(Timestep(None, LAST_ACTION, None, next_obs, None, {}))
    log(agent, T, LAST_ACTION, next_obs)
    T += 1

    pomdp_probs = agent.pomdp_probabilities
    mlp = pomdp_probs.argmax()
    mlpomdp = agent.pomdps[mlp]
    mls = agent.combined_belief.argmax()
    mlstate = mlpomdp.states[mls]

    belief_entropy = round(entropy(agent.beliefs[mlp], base=agent.pomdps[0].num_states), 2)

    terminal = belief_entropy < 0.10 and (mlstate[2:] == 2).all()
    if terminal:
        print("END")
        LAST_ACTION = 42
    else:
        LAST_ACTION = agent.action(None)

    #rospy.loginfo(f"Sending action {LAST_ACTION} to Manager node ({action_meanings[LAST_ACTION]})")
    send_manager_message(str(LAST_ACTION))

def setup_adhoc_agent():
    from balls_pomdp import load_task
    #rospy.loginfo("Loading pomdps")
    cache_directory = config["cache directory"]
    num_tasks = len(BALL_NODES)
    pomdps = []
    drawers = []
    for t in range(num_tasks):
        pomdp = load_task(t, cache_directory)
        pomdps.append(pomdp)
    agent = TEBOPA(pomdps, [pomdps[0].action_meanings.index("locate human")])
    agent.drawers = drawers
    try:
        from yaaf import rmdir
        rmdir("Estados mais prováveis")
    except:
        pass
    return agent


def log(agent, t, action, observation):

    pomdp_probs = agent.pomdp_probabilities

    mlp = pomdp_probs.argmax()
    mlpomdp = agent.pomdps[mlp]
    goal = BALL_NODES[mlp]

    mls = agent.combined_belief.argmax()
    mlstate = mlpomdp.states[mls]

    task_entropy = round(entropy(agent.pomdp_probabilities, base=len(agent.pomdp_probabilities)), 2)
    combined_entropy = round(agent.combined_entropy, 2)

    print(f"t: {t}", flush=True)
    print(f"Action: {mlpomdp.action_meanings[action]} (#{action})", flush=True)
    print(f"Observation: {observation} (#{mlpomdp.observation_index(observation)})", flush=True)
    print(f"Beliefs: {[round(prob, 5) for prob in agent.pomdp_probabilities]} ({task_entropy} entropy)", flush=True)
    print(f"MLPOMDP: {[node_names[node] for node in goal]} (#{mlp})", flush=True)
    print(f"MLS: {mlstate} (#{mls}) ({combined_entropy} entropy)\n", flush=True)

if __name__ == '__main__':

    opt = ArgumentParser()

    opt.add_argument("-node_name", default="adhoc_pomdp_decision")

    opt.add_argument("-manager_subscriber_topic", default="/adhoc_pomdp/manager_decision")
    opt.add_argument("-manager_publisher_topic", default="/adhoc_pomdp/decision_manager")

    opt.add_argument("-publisher_queue_size", type=int, default=100)
    opt.add_argument("-communication_refresh_rate", type=int, default=10)

    opt = opt.parse_args()

    rospy.init_node(opt.node_name)

    rospy.loginfo("Setting up algorithm and auxiliary structures")

    with open('../resources/config.yml') as config:
        config = yaml.load(config, Loader=yaml.FullLoader)
        node_names = [node[0] for node in config["manager"]["node descriptions"]]
        config = config["decision"]["gaips"]

      #####
    # Tasks #
      #####

    agent = setup_adhoc_agent()
    possible_tasks = [pomdp for pomdp in agent.pomdps]
    action_meanings = possible_tasks[-1].action_meanings

    rospy.loginfo(f"Initializing ROS Node {opt.node_name}")

    # ########### #
    # Subscribers #
    # ########### #

    rospy.loginfo(f"Setting up Manager node subscriber (local topic at {opt.manager_subscriber_topic})")
    manager_subscriber = rospy.Subscriber(opt.manager_subscriber_topic, String, receive_manager_message)

    # ########## #
    # Publishers #
    # ########## #

    rospy.loginfo(f"Setting up Manager node publisher (topic at {opt.manager_publisher_topic})")
    manager_publisher = rospy.Publisher(opt.manager_publisher_topic, String, queue_size=opt.publisher_queue_size)

    # ### #
    # Run #
    # ### #

    for t in reversed(range(5)):
        rospy.loginfo(f"Starting in {t+1}")
        time.sleep(1)

    LAST_ACTION = agent.action(None)
    T = 0

    rospy.loginfo(f"Sending action {LAST_ACTION} to Manager node ({action_meanings[LAST_ACTION]})")
    send_manager_message(str(LAST_ACTION))

    rospy.loginfo("Ready")
    rate = rospy.Rate(opt.communication_refresh_rate)
    rospy.spin()
