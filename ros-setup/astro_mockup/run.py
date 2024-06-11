"""
Astro Mockup
Final version will run on Astro (Python2)
"""
import time

import rospy
import yaml
from std_msgs.msg import String
from argparse import ArgumentParser

def receive_order_from_manager(message: String):
    order = message.data
    rospy.loginfo(f"Received Manager node order: '{order}'")
    if "go to" in order:
        new_x, new_y = order.replace(",", "").split(" ")[-2:]
        new_x, new_y = float(new_x), float(new_y)
        move_astro(new_x, new_y)
    x, y = read_current_coordinates()
    message = f"{x}, {y}"
    rospy.loginfo(f"After order new coordinates are {message}")
    send_manager_message(message)

def move_astro(new_x, new_y):
    rospy.loginfo(f"Astro is moving to {new_x}, {new_y}. Please wait...")
    for t in reversed(range(config["seconds to move"])):
        rospy.loginfo(f"{t + 1}")
        time.sleep(1)
    global x, y
    x, y = new_x, new_y

def read_current_coordinates():
    return x, y

def send_manager_message(message: str):
    rospy.loginfo(f"Sent Manager node message {message}")
    manager_publisher.publish(message)


if __name__ == '__main__':

    opt = ArgumentParser()

    opt.add_argument("-node_name", default="adhoc_pomdp_astro")

    opt.add_argument("-manager_subscriber_topic", default="/adhoc_pomdp/manager_astro")
    opt.add_argument("-manager_publisher_topic", default="/adhoc_pomdp/astro_manager")

    opt.add_argument("-publisher_queue_size", type=int, default=100)
    opt.add_argument("-communication_refresh_rate", type=int, default=1)
    opt.add_argument("-tts", action="store_true")

    opt = opt.parse_args()

    rospy.init_node(opt.node_name)

    rospy.loginfo(f"Initializing auxiliary structures")

    with open('../resources/config.yml') as config: config = yaml.load(config, Loader=yaml.FullLoader)["astro mockup"]

    # Initial position
    x, y = 0, 0

    rospy.loginfo(f"Initializing ROS Node {opt.node_name}")

    # ########### #
    # Subscribers #
    # ########### #

    rospy.loginfo(f"Setting up Manager node subscriber (local topic at {opt.manager_subscriber_topic})")
    callback = lambda message: receive_order_from_manager(message)
    manager_subscriber = rospy.Subscriber(opt.manager_subscriber_topic, String, callback)

    # ########## #
    # Publishers #
    # ########## #

    rospy.loginfo(f"Setting up Manager node publisher (topic at {opt.manager_publisher_topic})")
    manager_publisher = rospy.Publisher(opt.manager_publisher_topic, String, queue_size=opt.publisher_queue_size)

    # ### #
    # Run #
    # ### #

    rospy.loginfo("Ready")

    rate = rospy.Rate(opt.communication_refresh_rate)

    rospy.spin()
