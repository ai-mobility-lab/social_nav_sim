#!/usr/bin/env python

## import ros packages
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Twist
from gazebo_msgs.msg import ModelStates
from animated_obstacle_plugin.srv import *
from obstacle_manager.srv import *

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

class ObstacleManager():
    def __init__(self):
        # set parameters
        self.base_name = "obstacle_"

        # parameters
        ## parameters for controlling human
        self.env_radius = rospy.get_param("/environment/radius", 6.0)
        self.op_freq = rospy.get_param("/obstacle/operating_frequency", 10.0)
        self.is_env_in_sync = rospy.get_param("/obstacle/is_env_in_sync", False)
        self.env_time_step = rospy.get_param("/environment/time_step", 0.25)
        self.discomfort_dist = rospy.get_param("/environment/discomfort_dist", 0.25)
        self.num_obs = rospy.get_param("/obstacle/num_obstacle", 5)
        self.num_static = rospy.get_param("/obstacle/num_static", 0)
        if self.is_env_in_sync:
            self.act_freq = 1 / self.env_time_step
        else:
            self.act_freq = max(self.op_freq, 1 / self.env_time_step)
        self.policy = rospy.get_param("/obstacle/human/policy", "orca")
        self.FOV = rospy.get_param("/obstacle/human/FOV", 2.0)
        # a human may change its goal before it reaches its old goal
        self.random_goal_changing = rospy.get_param("/obstacle/human/random_goal_changing", True)
        self.goal_change_chance = rospy.get_param("/obstacle/human/goal_change_chance", 0.25)
        # a human may change its goal after it reaches its old goal
        self.end_goal_changing = rospy.get_param("/obstacle/human/end_goal_changing", True)
        self.end_goal_change_chance = rospy.get_param("/obstacle/human/end_goal_change_chance", 1.0)
        # a human may change its radius and/or v_pref after it reaches its current goal
        self.random_radii = rospy.get_param("/obstacle/human/random_radii", False)
        self.random_v_pref = rospy.get_param("/obstacle/human/random_v_pref", False)
        # one human may have a random chance to be blind to other agents at every time step
        self.random_unobservability = rospy.get_param("/obstacle/human/random_unobservability", False)
        self.unobservable_chance = rospy.get_param("/obstacle/human/unobservable_chance", 0.3)
        # randomly change policy
        self.random_policy_changing = rospy.get_param("/obstacle/human/random_policy_changing", False)
        # compute time_step
        self.dt = 1 / float(self.act_freq)
        self.obs_goal_arr = np.zeros([self.num_obs, 2])
        self.min_dist_arr = np.zeros([self.num_obs, self.num_obs])
        self.t0 = 0
        ## parameters for monitoring module
        self.is_initialized = False
        self.is_running = False

        # ROS SUBSCRIBERS
        ### pose subscriber
        self.sub_gazebo = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_model_callback, queue_size=1)

        # ROS SERVER
        self.srv_init = rospy.Service('/obstacle_manager/is_initialized', Init, self.init_srv_callback)
        self.srv_run = rospy.Service('/obstacle_manager/power_switch', Run, self.run_srv_callback)
        self.srv_reset = rospy.Service('/obstacle_manager/reset', Reset, self.reset_srv_callback)
        
        # initialize node
        self.reset()

    def _load_srv_cli(self):
        # check if gazebo is initialized
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        # ROS CLIENT
        self.cli_gazebo = {}
        self.cli_gazebo["set_vel"] = {}
        self.cli_gazebo["get_vel"] = {}
        for i in range(self.num_static, self.num_obs):
            try:
                cli_gazebo = rospy.ServiceProxy('/' + self.base_name + "%d"%i + '/SetActorVelocity', SetVel)
                self.cli_gazebo["set_vel"][i] = cli_gazebo
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
            try:
                cli_gazebo = rospy.ServiceProxy('/' + self.base_name + "%d"%i + '/GetActorVelocity', GetVel)
                self.cli_gazebo["get_vel"][i] = cli_gazebo
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
                
    def _initialize_obstacles(self):
        rospy.loginfo("[obstacle_manager] waiting for obstacle information to be set...")
        is_obs_loaded = False
        while not is_obs_loaded:
            if rospy.has_param("/obstacle_information"):
                is_obs_loaded = True
        self.obs_info = rospy.get_param("obstacle_information")
        self.obstacle = []
        for i in range(self.num_obs):
            # load and set config
            obs_dict = self.obs_info[i]
            if obs_dict["is_static"]:
                obs_dict["policy"] = "dummy"
            else:
                obs_dict["policy"] = self.policy
            obs_dict["policy_config"] = self.policy_config
            obs_dict["FOV"] = self.FOV
            obs_dict["time_step"] = self.dt

            # initialize an object
            obstacle = Obstacle(obs_dict)
            # store the intialized obstacle
            self.obstacle.append(obstacle)
            self.obs_info[i] = obs_dict
            
        # update goal_arr and min_dist_arr
        for i in range(self.num_obs):
            self.obs_goal_arr[i,0] = self.obstacle[i].gx
            self.obs_goal_arr[i,1] = self.obstacle[i].gy
            for j in range(self.num_obs):
                self.min_dist_arr[i,j] = self.obstacle[i].radius + self.obstacle[j].radius + self.discomfort_dist
        rospy.loginfo("[obstacle_manager] obstacle information loaded!")
    
    def _load_policy_param(self):
        self.policy_config = {}
        self.policy_config["time_step"] = self.dt
        if self.policy.lower() == "orca":
            self.policy_config["max_neighbors"] = self.num_obs
            self.policy_config["neighbor_dist"] = rospy.get_param("/obstacle/policy/orca/neighbor_dist", 10)
            self.policy_config["safety_space"] = rospy.get_param("/obstacle/policy/orca/safety_space", 0.15)
            self.policy_config["time_horizon"] = rospy.get_param("/obstacle/policy/orca/time_horizon", 5)
            self.policy_config["time_horizon_obst"] = rospy.get_param("/obstacle/policy/orca/time_horizon_obst", 5)
        elif self.policy.lower() == "social_force":
            self.policy_config["A"] = rospy.get_param("/obstacle/policy/social_force/A", 2)
            self.policy_config["B"] = rospy.get_param("/obstacle/policy/social_force/B", 1)
            self.policy_config["KI"] = rospy.get_param("/obstacle/policy/social_force/KI", 1)
    
    def _get_others_states(self, obs_id):
        observation = []
        for i in range(self.num_obs):
            if i != obs_id:
                observation_ = self.obstacle[i].get_observable_state()
                observation.append(observation_)
        return observation
    
    def _find_goal(self, obstacle, goal_arr, min_dist_list):
        # Produce valid goal for human in case of circle setting
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 1.0 if obstacle.v_pref == 0 else obstacle.v_pref
            noise = (np.random.random(2) - 0.5) * v_pref
            goal = self.env_radius * np.array([np.cos(angle), np.sin(angle)]) + noise
            collide = False
            
            dist = np.linalg.norm(goal - goal_arr, axis=1)
            if np.sum(dist < min_dist_list) > 0:
                collide = True
            if not collide:
                break
            
        return goal[0], goal[1]
    
    # Update the humans' end goals in the environment
    # Produces valid end goals for each human
    def _update_goals_randomly(self):
        # Update humans' goals randomly
        for idx, obstacle in enumerate(self.obstacle):
            if obstacle.is_static:
                continue
            
            goal_arr = self.obs_goal_arr.copy()
            goal_arr = np.vstack([goal_arr[:idx], goal_arr[idx+1:]])
            min_dist_list = np.append(self.min_dist_arr[idx,:idx], self.min_dist_arr[idx,idx+1:])
            if np.random.random() <= self.goal_change_chance:
                gx, gy = self._find_goal(obstacle, goal_arr, min_dist_list)
                # Give human new goal
                obstacle.set_goal(gx, gy)
                self.obs_goal_arr[idx,0] = gx
                self.obs_goal_arr[idx,1] = gy

    # Update the specified human's end goals in the environment randomly
    def _update_goal(self, obs_idx):
        obstacle = self.obstacle[obs_idx]
        goal_arr = self.obs_goal_arr.copy()
        goal_arr = np.vstack([goal_arr[:obs_idx], goal_arr[obs_idx+1:]])
        min_dist_list = np.append(self.min_dist_arr[obs_idx,:obs_idx], self.min_dist_arr[obs_idx,obs_idx+1:])
        # Update human's goals randomly
        if np.random.random() <= self.end_goal_change_chance:
            gx, gy = self._find_goal(obstacle, goal_arr, min_dist_list)
            # Give human new goal
            obstacle.set_goal(gx, gy)
            self.obs_goal_arr[obs_idx,0] = gx
            self.obs_goal_arr[obs_idx,1] = gy
    
    def reset(self, is_initial=True):
        if is_initial:
            self._load_policy_param()
        # obstacles must be initialized after pocliy parameters are set
        self._initialize_obstacles()
        if is_initial:
            self._load_srv_cli()
        self.is_initialized = True
        self.t0 = rospy.get_time()
        rospy.loginfo("[obstacle_manager] obstacle_manager node intialized")

    def gazebo_model_callback(self, msg):
        if not self.is_initialized:
            return
        for idx, name in enumerate(msg.name):
            if self.base_name in name:
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
                
    def init_srv_callback(self, req):
        return self.is_initialized
    
    def run_srv_callback(self, req):
        if self.is_initialized:
            if self.is_running:
                self.is_running = False
            else:
                self.is_running = True
        else:
            rospy.logerr("[obstacle_manager] obstacle_manager has not been initialized yet")
        return self.is_running
    
    def reset_srv_callback(self, req):
        self.is_running = False
        self.is_initialized = False
        self.reset(is_initial=False)
        return self.is_initialized

    def step(self):
        if self.is_running:
            vel_cmd = {}
            for i in range(self.num_static, self.num_obs):
                obstacle_ = self.obstacle[i]
                # don't compute actions of static obstacles
                # compute step action values of the current obstacle
                observation = self._get_others_states(i)
                # compute action values
                vx, vy = self.obstacle[i].act(observation)
                vel_cmd[i] = (vx, vy)
                
            for i in range(self.num_static, self.num_obs):
                self.cli_gazebo["set_vel"][i](*vel_cmd[i])
                
            # Update all humans' goals randomly midway through episode
            if self.random_goal_changing:
                t1 = rospy.get_time()
                if t1 - self.t0 > 5.0:
                    self._update_goals_randomly()
                    self.t0 = t1

            # Update a specific human's goal once its reached its original goal
            if self.end_goal_changing:
                for idx, obstacle in enumerate(self.obstacle):
                    if not obstacle.is_static and obstacle.is_goal_reached():
                        self._update_goal(idx)
        else:
            for i in range(self.num_obs):
                # halt obstacles
                if abs(self.obstacle[i].vx) + abs(self.obstacle[i].vy) > 0:
                    self.cli_gazebo["set_vel"][i](0, 0)
            self.t0 = rospy.get_time()

def main():
    rospy.init_node('obstacle_manager_node')
    obstacle_manager_node = ObstacleManager()

    rate = rospy.Rate(obstacle_manager_node.op_freq)

    while not rospy.is_shutdown():
        if obstacle_manager_node.is_initialized:
            obstacle_manager_node.step()
        rate.sleep()

if __name__ == '__main__':
    main()