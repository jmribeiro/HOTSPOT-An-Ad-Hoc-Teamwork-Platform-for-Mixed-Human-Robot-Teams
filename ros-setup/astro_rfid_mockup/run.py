"""
Astro RFID Mockup
Final version will run on Astro (Python2)
"""

import rospy
from std_msgs.msg import String
from argparse import ArgumentParser

if __name__ == '__main__':

    opt = ArgumentParser()

    opt.add_argument("-node_name", default="adhoc_pomdp_astro_rfid")

    opt.add_argument("-manager_publisher_topic", default="/idmind_rfid")

    opt.add_argument("-publisher_queue_size", type=int, default=100)
    opt.add_argument("-communication_refresh_rate", type=int, default=1)
    opt.add_argument("-tts", action="store_true")

    opt = opt.parse_args()

    rospy.init_node(opt.node_name)
    rospy.loginfo(f"Initializing ROS Node {opt.node_name}")

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

    prompt = ""
    while prompt != "exit":
        prompt = input("Insert RFID > ")
        manager_publisher.publish(str(prompt))
