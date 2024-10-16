#! /usr/bin/env python
# Import ROS.
import rospy
# Importing BoundingBoxes message from package darknet_ros_msgs.
from darknet_ros_msgs.msg import BoundingBoxes
# Import the API.
from pymavlink import mavutil
import time
import numpy as np
from PER_DDPG import DDPG as PER_DDPG
import PER_buffer
from utils import LinearSchedule
from env import Env

mode_g = False
cx=0
cy=0

MAX_EPISODES = 150
MAX_EP_STEPS = 100
LR_A = 0.002  # learning rate for actor
LR_C = 0.004    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
Action_dim=2
Action_max=0.1
STATE_dim=6

def move_drone(drone, speeds):
    speed_y,speed=speeds

    drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,drone.target_system,drone.target_component,
                      mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,int(0b010111000111),0,0,0,speed_y,0,0,0,0,0,0,speed))
    time.sleep(0.5)
    drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,drone.target_system,drone.target_component,
                      mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,int(0b010111000111),0,0,0,0,0,0,0,0,0,0,0)) 
    time.sleep(6)
    
def stop_drone(drone):
    drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,drone.target_system,drone.target_component,
                      mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,int(0b110111000111),0,0,0,0,0,0,0,0,0,0,0))        
    
def detection_cb(msg):
    # Callback function of the subscriber.
    # Assigning bounding_boxes of the message to bbox variable.
    bbox = msg.bounding_boxes
    global cx
    global cy
    global mode_g
    cx=0
    cy=0
    #print("The value of length is: ",len(bbox))
    for i in range(len(bbox)):
        #print("Bounding Box:")
        #print(bbox)
        # Printing the detected object with its probability in percentage.
        #rospy.loginfo("{}% certain {} detected.".format(
            #float(bbox[i].probability * 100), bbox[i].Class))
        if bbox[i].Class == "person":
            print("Person Detected")
            cx=bbox[i].xmin+(bbox[i].xmax-bbox[i].xmin)//2
            cy=bbox[i].ymin+(bbox[i].ymax-bbox[i].ymin)//2
            area=(bbox[i].xmax-bbox[i].xmin)*(bbox[i].ymax-bbox[i].ymin)         
            # Setting mode_g to True to mark presence of person.
            mode_g = True
            #rospy.loginfo("Person found. Starting Rescue Operation")
        else:
            mode_g=False
            
def wait4start(vehicle):
    # Wait a heartbeat before sending commands
    vehicle.wait_heartbeat()
    while True:
        msg =vehicle.recv_match(type = 'HEARTBEAT', blocking = False)
        if msg:
            mode = mavutil.mode_string_v10(msg)
            print("Waiting for Guided")
            time.sleep(1)
            if mode=='GUIDED':
                print("Guided Mode")
                break


def main():
    global cx
    global cy
    global mode_g
    # Initialise ROS node
    rospy.init_node("search_and_rescue_node")
    # Creating a subscriber for the topic "/darknet_ros/bounding_boxes".
    rospy.Subscriber(name="/darknet_ros/bounding_boxes",
                     data_class=BoundingBoxes,
                     callback=detection_cb,
                     queue_size=1)
    # Create an object for the API.
    drone=mavutil.mavlink_connection('udpin:localhost:14550')
    drone.wait_heartbeat()
    print("Heartbeat from system (system %u component %u)" %
      (drone.target_system, drone.target_component))
    
    #---------------------------------Initializing PER-DDPG POLICY-----------------------------------------------------
    #PER parameters
    prioritized_replay_alpha=0.6
    prioritized_replay_beta0=0.4
    prioritized_replay_beta_iters=None
    prioritized_replay_eps=10000
    #Create DDPG policy_per_ddpg
    policy_per_ddpg = PER_DDPG(STATE_dim, Action_dim, Action_max)
    #If we are not doing prioritized experience replay
    #Then we use my implementation of the uniform replay buffe
    replay_buffer = PER_buffer.PrioritizedReplayBuffer(MEMORY_CAPACITY, prioritized_replay_alpha)
    if prioritized_replay_beta_iters is None:
        prioritized_replay_beta_iters = MAX_EP_STEPS*MAX_EPISODES
    #Create annealing schedule
    beta_schedule = LinearSchedule(prioritized_replay_beta_iters, initial_p=prioritized_replay_beta0, final_p=1.0)
    #--------------------------------------------------------------------------------------------------------------------
    myenv=Env(MAX_EP_STEPS,STATE_dim)
    episode_rewardall=[]
    state_all=[]

    for i in range(MAX_EPISODES):
        episode_r=0
        input("Enter When Bounding Box is appearing...")
        print(f"Starting Episode {i}.")
        state=myenv.reset([350,240],[cx,cy])
        done=False
        prev_state=[0,0,0,0,0,0]
        var=1
        while done!=True:

            action=policy_per_ddpg.get_action(np.array(state))
            print(f"State: {state}, action: {action}, mode_g: {mode_g}")
            action[0] = np.clip(np.random.normal(action[0], var), 0, 1)
            action[1] = np.clip(np.random.normal(action[1], var), 0, 1)
            move_drone(drone,action)
            reward,next_state,done=myenv.step(prev_state,state,action,[350,240],[cx,cy],not(mode_g))
            episode_r+=reward
            replay_buffer.add(state,action,[reward],next_state,done)
            if var > 0.1:
                    var *= .9998
            
            beta_value = 0
            beta_value = beta_schedule.value(i)
        #print("---TRAINING PER-DDPG---")
            policy_per_ddpg.train(replay_buffer, True, beta_value, prioritized_replay_eps,BATCH_SIZE, GAMMA,TAU)

            if done:
                print(f"Episode Reward: {episode_r}")
                episode_rewardall.append(episode_r)
                stop_drone(drone)
            state_all.append(state)
            prev_state=state
            state=next_state
            
    rospy.signal_shutdown()


# Driver code.
if __name__ == '__main__':
    try:
        main()
        # Used to keep the node running.
        rospy.spin()
    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        exit()