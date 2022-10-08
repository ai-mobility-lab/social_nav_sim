#!/usr/bin/env python

## import ros packages
import rospy
import rospkg
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from animated_obstacle_plugin.srv import *
from agent.srv import *

## import fundamental packages
import numpy as np
import os, sys
from pathlib import Path
import math

# add custom libraries
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils import *
from state.obstacle import Obstacle
from state.robot import Robot

class Agent():
    def __init__(self):
        rospack = rospkg.RosPack()
        self.pkg_root = rospack.get_path("agent")
        # set parameters
        self.agent_name = "agent"
        self.obs_name = "obstacle_"
        self.urdf = rospy.get_param("/robot_description")

        # parameters
        ## parameters for setting environment
        self.env_radius = rospy.get_param("/environment/radius", 6.0)
        self.env_time_step = rospy.get_param("/environment/time_step", 0.25)
        self.discomfort_dist = rospy.get_param("/environment/discomfort_dist", 0.25)
        self.num_obs = rospy.get_param("/obstacle/num_obstacle", 5)
        self.num_static = rospy.get_param("/obstacle/num_static", 0)
        self.freq = 1 / self.env_time_step
        ## parameters for controlling robot
        self.radius = rospy.get_param("/agent/robot/radius", 0.3)
        self.v_pref = rospy.get_param("/agent/robot/v_pref", 1.0)
        self.FOV = rospy.get_param("/agent/robot/FOV", 2.0)
        self.kinematics = rospy.get_param("/agent/robot/kinematics", "holonomic")
        self.policy = rospy.get_param("/agent/robot/policy", "srnn")
        # compute time_step
        self.dt = 1 / float(self.freq)
        self.t0 = 0
        ## parameters for monitoring module
        self.is_initialized = False
        self.is_running = False

        # ROS SUBSCRIBERS
        ### pose subscriber
        self.sub_gazebo = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_model_callback, queue_size=1)
        self.sub_pose = rospy.Subscriber("/humic/ground_truth_pose", Odometry, self.robot_pose_callback, queue_size=1)

        # ROS PUBLISHERS
        self.pub_cmd_vel = rospy.Publisher("/humic/cmd_vel", Twist, queue_size=1)
        
        # ROS SERVER
        self.srv_init = rospy.Service('/agent/is_initialized', Init, self.init_srv_callback)
        self.srv_run = rospy.Service('/agent/power_switch', Run, self.run_srv_callback)
        self.srv_reset = rospy.Service('/agent/reset', Reset, self.reset_srv_callback)

        # initialize agent module
        self.reset()

    def _load_srv_cli(self):
        # check if gazebo is initialized
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        self.cli_gazebo = {}
        # ROS CLIENT
        self.cli_gazebo["get_vel"] = {}
        for i in range(self.num_static, self.num_obs):
            try:
                cli_gazebo = rospy.ServiceProxy('/' + self.obs_name + "%d"%i + '/GetActorVelocity', GetVel)
                self.cli_gazebo["get_vel"][i] = cli_gazebo
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

    def _get_obstacle_info(self):
        rospy.loginfo("[agent_node] waiting for obstacle information to be set...")
        is_obs_loaded = False
        while not is_obs_loaded:
            if rospy.has_param("/obstacle_information"):
                is_obs_loaded = True
        # get obstacle information
        self.obs_info = rospy.get_param("/obstacle_information")
        self.obstacle = []
        for i in range(self.num_obs):
            # load and set config
            obs_dict = self.obs_info[i]
            # initialize an object
            obstacle = Obstacle(obs_dict)
            # store the intialized obstacle
            self.obstacle.append(obstacle)
            self.obs_info[i] = obs_dict
        rospy.loginfo("[agent_node] obstacle information loaded!")

    def _load_policy_param(self):
        self.policy_config = {}
        self.policy_config["time_step"] = self.dt
        if self.policy.lower() == "orca":
            self.policy_config["max_neighbors"] = self.num_obs
            self.policy_config["neighbor_dist"] = rospy.get_param("/agent/policy/orca/neighbor_dist", 10)
            self.policy_config["safety_space"] = rospy.get_param("/agent/policy/orca/safety_space", 0.15)
            self.policy_config["time_horizon"] = rospy.get_param("/agent/policy/orca/time_horizon", 5)
            self.policy_config["time_horizon_obst"] = rospy.get_param("/agent/policy/orca/time_horizon_obst", 5)
        elif self.policy.lower() == "social_force":
            self.policy_config["A"] = rospy.get_param("/agent/policy/social_force/A", 2)
            self.policy_config["B"] = rospy.get_param("/agent/policy/social_force/B", 1)
            self.policy_config["KI"] = rospy.get_param("/agent/policy/social_force/KI", 1)
        elif self.policy.lower() == "srnn":
            self.policy_config["device"] = rospy.get_param("/agent/policy/srnn/device", "cuda")
            self.policy_config["coord_frame"] = rospy.get_param("/agent/policy/srnn/coord_frame", "absolute")
            ckpt_file = rospy.get_param("/agent/policy/srnn/ckpt_file", "5d_abs_holo.pt")
            self.policy_config["ckpt_path"] = os.path.join(self.pkg_root, "param/ckpts", ckpt_file)
            self.policy_config["torch_seed"] = rospy.get_param("/agent/policy/srnn/torch_seed", 0)
            self.policy_config["action_size"] = rospy.get_param("/agent/policy/srnn/action_size", 2)
            self.policy_config["num_obstacle"] = self.num_obs
            # RNN size
            self.policy_config["human_node_rnn_size"] = rospy.get_param("/agent/policy/srnn/human_node_rnn_size", 128)
            self.policy_config["human_human_edge_rnn_size"] = rospy.get_param("/agent/policy/srnn/human_human_edge_rnn_size", 256)
            # Input and output size
            self.policy_config["human_node_input_size"] = rospy.get_param("/agent/policy/srnn/human_node_input_size", 3)
            self.policy_config["human_human_edge_input_size"] = rospy.get_param("/agent/policy/srnn/human_human_edge_input_size", 2)
            self.policy_config["human_node_output_size"] = rospy.get_param("/agent/policy/srnn/human_node_output_size", 256)
            self.policy_config["robot_state_size"] = rospy.get_param("/agent/policy/srnn/robot_state_size", 7)
            # Embedding size
            self.policy_config["human_node_embedding_size"] = rospy.get_param("/agent/policy/srnn/human_node_embedding_size", 64)
            self.policy_config["human_human_edge_embedding_size"] = rospy.get_param("/agent/policy/srnn/human_human_edge_embedding_size", 64)
            # Attention vector dimension
            self.policy_config["attention_size"] = rospy.get_param("/agent/policy/srnn/attention_size", 64)

    def _init_robot(self):
        rospy.loginfo("[agent_node] waiting for agent information to be set...")
        is_agent_loaded = False
        while not is_agent_loaded:
            if rospy.has_param("/agent_information"):
                is_agent_loaded = True
        # get obstacle information
        self.agent_info = rospy.get_param("/agent_information")
        # load and set config
        self.agent_info["size"] = self.radius
        self.agent_info["v_pref"] = self.v_pref
        self.agent_info["kinematics"] = self.kinematics
        self.agent_info["policy"] = self.policy
        self.agent_info["policy_config"] = self.policy_config
        self.agent_info["FOV"] = self.FOV
        self.agent_info["time_step"] = self.dt

        # initialize an object
        self.robot = Robot(self.agent_info)
        rospy.loginfo("[agent_node] agent information loaded!")

        self.is_initialized = True

    # generate observation for each timestep
    def _generate_ob(self):
        if self.policy == "srnn":
            observation = {}
            # nodes
            if self.policy_config["coord_frame"] == "absolute":
                observation['robot_node'] = self.robot.get_full_state_list_noV()
            else:
                observation['robot_node'] = self.robot.get_full_relative_state_list_noV()

            # edges
            # temporal edge: robot's velocity
            observation["temporal_edges"] = np.array(self.robot.last_action)

            # spatial edges: the vector pointing from the robot position to each human's position
            observation['spatial_edges'] = np.zeros((self.num_obs, 2))
            for i in range(self.num_obs):
                obstacle_ = self.obstacle[i]
                relative_pos = self.robot.get_rel_pos(obstacle_.px, obstacle_.py)
                observation['spatial_edges'][i] = relative_pos
        else:
            observation = []
            for i in range(self.num_obs):
                observation_ = self.obstacle[i].get_observable_state()
                observation.append(observation_)

        return observation

    def _clip_action(self, action):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        if isinstance(action, tuple):
            action = list(action)
        # clip the action
        if self.kinematics == "unicycle":
            # raw_action[0] = np.clip(raw_action[0], -0.1, 0.1) # action[0] is change of v
            action[0] = np.clip(action[0], -0.25, 0.25) # action[0] is change of v
            # raw_action[1] = np.clip(raw_action[1], -0.1, 0.1) # action[1] is change of theta
            action[1] = np.clip(action[1], -0.25, 0.25) # action[1] is change of theta

            return action
        else:
            act_norm = np.linalg.norm(np.array(action))
            if act_norm > self.v_pref:
                action[0] = action[0] / act_norm * self.v_pref
                action[1] = action[1] / act_norm * self.v_pref
            return action
            
    def reset(self, is_initial=True):
        self._get_obstacle_info()
        if is_initial:
            self._load_policy_param()
            self._load_srv_cli()
        # robot must be initialized after pocliy parameters are set
        self._init_robot()
        self.is_initialized = True
        self.t0 = rospy.get_time()
        rospy.loginfo("[agent] agent node intialized")

    def gazebo_model_callback(self, msg):
        if not self.is_initialized:
            return
        for idx, name in enumerate(msg.name):
            if self.obs_name in name:
                # check id 
                obs_id = int(name.split("_")[-1])
                pose = msg.pose[idx]
                # update obstacle's state
                pos_xy = vector3_to_array(pose.position)[:2]
                theta = euler_from_quaternion(quat_to_array(pose.orientation))[2]
                if self.obstacle[obs_id].is_static:
                    vel_xy = [0, 0]
                else:
                    vel_msg = self.cli_gazebo["get_vel"][obs_id]()
                    vel_xy = [vel_msg.x, vel_msg.y]
                self.obstacle[obs_id].update(pos_xy, vel_xy, theta)

    def robot_pose_callback(self, msg):
        if not self.is_initialized:
            return
        pose = msg.pose.pose
        vel = msg.twist.twist
        # update robot's state
        pos_xy = vector3_to_array(pose.position)[:2]
        vel_xy = vector3_to_array(vel.linear)[:2]
        theta = euler_from_quaternion(quat_to_array(pose.orientation))[2]
        self.robot.update(pos_xy, vel_xy, theta)
        
    def init_srv_callback(self, req):
        return self.is_initialized
    
    def run_srv_callback(self, req):
        if self.is_initialized:
            if self.is_running:
                self.is_running = False
            else:
                self.is_running = True
        else:
            rospy.logerr("[agent] agent_node has not been initialized yet")
        return self.is_running
    
    def reset_srv_callback(self, req):
        self.is_running = False
        self.is_initialized = False
        self.reset(is_initial=False)
        return self.is_initialized

    def step(self):
        # compute difference between python env and gazebo sim to estimate noise level
        # dp = np.array([self.robot.px - self.robot.last_pos[0], self.robot.py - self.robot.last_pos[1]])
        # dp_norm = np.linalg.norm(dp)
        # dtheta = 1 - np.cos(self.robot.theta - self.robot.last_theta)
        # rospy.loginfo("[agent] error_p: %f, error_theta: %f"%(dp_norm, dtheta))
        # initialize a twist message to be published
        cmd_vel = Twist()
        if self.is_running:
            # arrange observation
            observation = self._generate_ob()
            # compute action value of robot
            action = self.robot.act(observation)
            action = self._clip_action(action)
            if self.kinematics == "unicycle":
                d_vx, d_yaw = action
                # add d_vx and clip out resulted velocity
                vx_b = np.clip(self.robot.vx_b + d_vx, -self.v_pref, self.v_pref)
                # set robot cmd_vel
                cmd_vel.linear.x = vx_b
                cmd_vel.angular.z = d_yaw / self.dt
                # store last action
                self.robot.set_last_action([vx_b, d_yaw])
            else:
                vx_w, vy_w = action
                yaw_tgt = np.arctan2(vy_w, vx_w)
                d_yaw = yaw_tgt - self.robot.theta
                # convert the reference frame of velocity from world to body
                vx_b, vy_b = self.robot.get_vel_in_body(vx_w, vy_w)
                # set robot cmd_vel
                cmd_vel.linear.x = vx_b
                cmd_vel.linear.y = vy_b
                cmd_vel.angular.z = d_yaw / self.dt
                # store last action
                self.robot.set_last_action([vx_w, vy_w])
            # publish robot cmd_vel
            self.pub_cmd_vel.publish(cmd_vel)
        else:
            if abs(self.robot.vx) + abs(self.robot.vy) > 0:
                # publish zero cmd_vel
                self.pub_cmd_vel.publish(cmd_vel)
        
def main():
    rospy.init_node('agent_node')
    agent_node = Agent()

    rate = rospy.Rate(agent_node.freq)

    while not rospy.is_shutdown():
        if agent_node.is_initialized:
            agent_node.step()
        rate.sleep()

if __name__ == '__main__':
    main()