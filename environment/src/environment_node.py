#!/usr/bin/env python

## import ros packages
import rospy
import rospkg
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelState, ModelStates
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnModel, SetModelState, DeleteModel
from animated_obstacle_plugin.srv import *
from obstacle_manager.srv import Init as ObsMgrInit
from obstacle_manager.srv import Run as ObsMgrRun
from obstacle_manager.srv import Reset as ObsMgrReset
from agent.srv import Init as AgentInit
from agent.srv import Run as AgentRun
from agent.srv import Reset as AgentReset
from environment.srv import Run, Reset

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
from state.model import Model

class Environment():
    def __init__(self):
        # set parameters
        self.agent_name = "agent"
        self.obs_name = "obstacle_"
        self.goal_name = "goal"
        self.urdf = rospy.get_param("/robot_description")
        
        # parameters
        ## parameters for setting environment
        self.env_radius = rospy.get_param("/environment/radius", 6.0)
        self.env_time_limit = rospy.get_param("/environment/time_limit", 50)
        self.env_time_step = rospy.get_param("/environment/time_step", 0.25)
        self.freq = rospy.get_param("/environment/monitor_frequency", 100)
        self.discomfort_dist = rospy.get_param("/environment/discomfort_dist", 0.25)
        self.num_obs = rospy.get_param("/obstacle/num_obstacle", 5)
        self.num_static = rospy.get_param("/obstacle/num_static", 0)
        self.robot_v_pref = rospy.get_param("/agent/robot/v_pref", 1.0)
        self.robot_radius = rospy.get_param("/agent/robot/radius", 0.3)
        self.auto_start = rospy.get_param("/environment/auto_start", False)
        self.auto_termination = rospy.get_param("/environment/auto_termination", False)
        ## fine model path to goal marker
        ros_pkg_finder = rospkg.RosPack()
        ros_pkg_path = ros_pkg_finder.get_path('assets')
        self.goal_marker_path = os.path.join(ros_pkg_path, "models", "goal_marker", "model.sdf")
        # compute time_step
        self.t0 = 0
        self.t0_timeout = 0
        ## parameters for monitoring module
        self.is_initialized = False
        self.is_running = False
        self.is_initial_idle = True
        self.episode_stat = {"total":0, "success":0, "collision":0, "timeout":0}
        self.num_goal_respawned = 0
        ## for checking collision
        self.obs_pos_arr = np.zeros([self.num_obs, 2])
        self.min_dist_arr = np.zeros(self.num_obs)

        # ROS SUBSCRIBERS
        ### pose subscriber
        self.sub_gazebo = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_model_callback, queue_size=1)
        self.sub_pose = rospy.Subscriber("/humic/ground_truth_pose", Odometry, self.robot_pose_callback, queue_size=1)
        
        # ROS SERVER
        self.srv_run = rospy.Service('/environment/power_switch', Run, self.run_srv_callback)
        self.srv_reset = rospy.Service('/environment/reset', Reset, self.reset_srv_callback)

        # initialize agent module
        self.reset()
        
    def _load_srv_cli(self):
        # check if gazebo is initialized
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        # ROS CLIENT
        self.cli_gazebo = {}
        self.cli_gazebo["set_pose"] = {}
        self.cli_gazebo["get_vel"] = {}
        self.cli_gazebo["init"] = {}
        for i in range(self.num_obs):
            obs_dict = self.obs_info[i]
            if obs_dict["is_human"]:
                try:
                    cli_gazebo = rospy.ServiceProxy('/' + self.obs_name + "%d"%i + '/SetActorPosition', SetPose)
                    self.cli_gazebo["set_pose"][i] = cli_gazebo
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)
                try:
                    cli_gazebo = rospy.ServiceProxy('/' + self.obs_name + "%d"%i + '/GetActorVelocity', GetVel)
                    self.cli_gazebo["get_vel"][i] = cli_gazebo
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)
                try:
                    cli_gazebo = rospy.ServiceProxy('/' + self.obs_name + "%d"%i + '/InitializeActor', Init)
                    self.cli_gazebo["init"][i] = cli_gazebo
                except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)
        try:
            cli_gazebo = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            self.cli_gazebo["set_model"] = cli_gazebo
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        try:
            self.cli_gazebo["spawn_urdf_model"] = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        try:
            self.cli_gazebo["spawn_sdf_model"] = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        try:
            self.cli_gazebo["delete_model"] = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
                
        self.cli_models = {}
        self.cli_models["agent"] = {}
        self.cli_models["obstacle"] = {}
        try:
            cli_model = rospy.ServiceProxy('/agent/is_initialized', AgentInit)
            self.cli_models["agent"]["init"] = cli_model
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        try:
            cli_model = rospy.ServiceProxy('/agent/power_switch', AgentRun)
            self.cli_models["agent"]["run"] = cli_model
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        try:
            cli_model = rospy.ServiceProxy('/agent/reset', AgentReset)
            self.cli_models["agent"]["reset"] = cli_model
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        try:
            cli_model = rospy.ServiceProxy('/obstacle_manager/is_initialized', ObsMgrInit)
            self.cli_models["obstacle"]["init"] = cli_model
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        try:
            cli_model = rospy.ServiceProxy('/obstacle_manager/power_switch', ObsMgrRun)
            self.cli_models["obstacle"]["run"] = cli_model
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        try:
            cli_model = rospy.ServiceProxy('/obstacle_manager/reset', ObsMgrReset)
            self.cli_models["obstacle"]["reset"] = cli_model
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def _init_obstacles(self):
        rospy.loginfo("[environment] waiting for obstacle information to be set...")
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
            obstacle = Model(obs_dict)
            # store the intialized obstacle
            self.obstacle.append(obstacle)
            self.obs_info[i] = obs_dict
        rospy.loginfo("[environment] obstacle information loaded!")

    def _init_robot(self):
        rospy.loginfo("[environment] waiting for agent information to be set...")
        is_agent_loaded = False
        while not is_agent_loaded:
            if rospy.has_param("/agent_information"):
                is_agent_loaded = True
        # get obstacle information
        self.agent_info = rospy.get_param("/agent_information")
        # load and set config
        self.agent_info["size"] = self.robot_radius
        self.agent_info["v_pref"] = self.robot_v_pref

        # initialize an object
        self.robot = Model(self.agent_info)
        rospy.loginfo("[environment] agent information loaded!")

    def _spawn_robot(self):
        # spawn robot urdf model in gazebo
        initial_pose = Pose()
        initial_pose.position.x = self.robot.px
        initial_pose.position.y = self.robot.py
        initial_pose.position.z = 0
        quat = quaternion_from_euler(0, 0, self.robot.theta)
        initial_pose.orientation.x = quat[0]
        initial_pose.orientation.y = quat[1]
        initial_pose.orientation.z = quat[2]
        initial_pose.orientation.w = quat[3]

        self.cli_gazebo["spawn_urdf_model"](
            model_name=self.agent_name,
            model_xml=self.urdf,
            robot_namespace='',
            initial_pose=initial_pose,
            reference_frame='map'
            )
        
    def _compute_collision_arr(self):
        # get object pose array
        for i in range(self.num_obs):
            self.obs_pos_arr[i,:] = [self.obstacle[i].px, self.obstacle[i].py]
            self.min_dist_arr[i] = self.obstacle[i].radius + self.robot.radius
            
    def _relocate_models(self):
        # reinitialize models' positions
        agent_info, obs_info = initialize_model_locations(self.obs_info)
        # set rosparam for publishing agent information
        rospy.set_param("agent_information", agent_info)
        # set rosparam for publishing obstacle information
        rospy.set_param("obstacle_information", obs_info)
        # get human model's default height
        actor_default_height = rospy.get_param("/obstacle/visual/actor_default_height")
        
        # relocate obstacle models
        for i in range(self.num_obs):
            obs_dict = obs_info[i]
            if obs_dict["is_human"]:
                self.cli_gazebo["set_pose"][i](obs_dict["init_pos"][0],obs_dict["init_pos"][1])
                res = self.cli_gazebo["init"][i]()
                if not res.is_initialized:
                    rospy.logerr("[environment] %s%d orientation not initialized"%(self.obs_name, i))
            else:
                msg = ModelState()
                msg.model_name = self.obs_name + "%d"%i
                init_pos = [obs_dict["init_pos"][0],obs_dict["init_pos"][1],0]
                msg.pose = array_to_posmsg(init_pos, [0, 0, 0, 1])
                self.cli_gazebo["set_model"](msg)
        # relocate agent
        msg = ModelState()
        msg.model_name = self.agent_name
        init_pos = [agent_info["init_pos"][0], agent_info["init_pos"][1],0]
        quat = quaternion_from_euler(0, 0, np.pi + np.arctan2(agent_info["init_pos"][1], agent_info["init_pos"][0]))
        msg.pose = array_to_posmsg(init_pos, quat)
        self.cli_gazebo["set_model"](msg)
        # relocate goal
        ### the goal_marker model is static, so it can't be moved
        ### just delete the marker and respawn it at a new location
        ## delete the goal marker
        self.cli_gazebo["delete_model"](self.goal_name)
        self.num_goal_respawned += 1
        self.goal_name = "goal%d"%self.num_goal_respawned
        ## spawn a new goal marker
        initial_pose = Pose()
        initial_pose.position.x = agent_info["goal"][0]
        initial_pose.position.y = agent_info["goal"][1]
        initial_pose.position.z = 0
        initial_pose.orientation.x = 0
        initial_pose.orientation.y = 0
        initial_pose.orientation.z = 0
        initial_pose.orientation.w = 1
        self.cli_gazebo["spawn_sdf_model"](
            model_name=self.goal_name,
            model_xml=open(self.goal_marker_path, 'r').read(),
            robot_namespace='',
            initial_pose=initial_pose,
            reference_frame='map'
            )

    def _check_collision(self):
        # compute distance
        dist = np.linalg.norm(np.array([self.robot.px, self.robot.py]) - self.obs_pos_arr, axis=1)
   
        # check if collision occurred
        is_collision = np.sum(dist < self.min_dist_arr) > 0
        
        if is_collision:
            col_id = np.argmax(is_collision)
            rospy.logerr("[environment] robot at (%f,%f) has collision with obstacle_%d at (%f,%f)"%(self.robot.px, self.robot.py, col_id, self.obstacle[col_id].px, self.obstacle[col_id].py))
            rospy.logwarn("[environment] COLLISION!")
            
        return is_collision
    
    def _check_success(self):
        is_success = self.robot.is_goal_reached()
        if is_success:
            rospy.logwarn("[environment] SUCCESS!")
        return is_success
    
    def _check_timeout(self):
        is_timeout = False
        t1 = rospy.get_time()
        if (t1 - self.t0) > self.env_time_limit:
            is_timeout = True
            
        if is_timeout:
            rospy.logwarn("[environment] TIME OUT!")
            
        return is_timeout
            
    def reset(self, is_initial=True):
        self._init_obstacles()
        self._init_robot()
        self._compute_collision_arr()
        if is_initial:
            self._load_srv_cli()
            self._spawn_robot()
        self.is_initialized = True
        rospy.loginfo("[environment] environment node intialized")

    def gazebo_model_callback(self, msg):
        if self.is_initialized is False:
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
                self.obs_pos_arr[obs_id,:] = pos_xy

    def robot_pose_callback(self, msg):
        if self.is_initialized is False:
            return
        pose = msg.pose.pose
        vel = msg.twist.twist
        # update robot's state
        pos_xy = vector3_to_array(pose.position)[:2]
        vel_xy = vector3_to_array(vel.linear)[:2]
        theta = euler_from_quaternion(quat_to_array(pose.orientation))[2]
        self.robot.update(pos_xy, vel_xy, theta)
        
    def run_srv_callback(self, req):
        if self.is_initialized:
            if self.is_running == True:
                res_agent = self.cli_models["agent"]["run"]()
                res_obstacle = self.cli_models["obstacle"]["run"]()

                if res_agent.is_running == False:
                    rospy.loginfo("[environment] agent node succefully powered OFF")
                else:
                    rospy.loginfo("[environment] agent node not powered OFF")
                    return self.is_running
                
                if res_obstacle.is_running == False:
                    rospy.loginfo("[environment] obstacle_manager node succefully powered OFF")
                else:
                    rospy.loginfo("[environment] obstacle_manager node not powered OFF")
                    return self.is_running
                
                self.is_running = False
                self.t0_timeout = rospy.get_time()
                rospy.loginfo("[environment] environment node succefully powered OFF")
            else:
                res_agent = self.cli_models["agent"]["init"]()
                res_obstacle = self.cli_models["obstacle"]["init"]()
                
                if res_agent.is_initialized == False:
                    rospy.loginfo("[environment] agent node not initialized")
                    return self.is_running
                
                if res_obstacle.is_initialized == False:
                    rospy.loginfo("[environment] obstacle_manager node not initialized")
                    return self.is_running
                
                res_agent = self.cli_models["agent"]["run"]()
                res_obstacle = self.cli_models["obstacle"]["run"]()
                
                if res_agent.is_running:
                    rospy.loginfo("[environment] agent node succefully powered ON")
                else:
                    rospy.loginfo("[environment] agent node not powered ON")
                    return self.is_running
                
                if res_obstacle.is_running:
                    rospy.loginfo("[environment] obstacle_manager node succefully powered ON")
                else:
                    rospy.loginfo("[environment] obstacle_manager node not powered ON")
                    return self.is_running
                
                self.is_running = True
                if self.is_initial_idle:
                    self.t0 = rospy.get_time()
                    self.is_initial_idle = False
                    rospy.logwarn("[environment] agent's goal: (%f, %f)"%(self.robot.gx, self.robot.gy))
                else:
                    self.t0 += (rospy.get_time() - self.t0_timeout)
                
                rospy.loginfo("[environment] environment node succefully powered ON")
        else:
            rospy.logerr("[environment] environment has not been initialized yet")
        return self.is_running
    
    def reset_srv_callback(self, req):
        self.is_running = False
        self.is_initial_idle = True
        self.is_initialized = False
        self._relocate_models()
        self.reset(is_initial=False)
        if self.is_initialized:
            res = self.cli_models["agent"]["reset"]()
            if res.is_initialized:
                rospy.loginfo("[environment] agent succefully reset")
            else:
                rospy.loginfo("[environment] agent not reset")
                self.is_initialized = False
                return self.is_initialized
            res = self.cli_models["obstacle"]["reset"]() 
            if res.is_initialized:
                rospy.loginfo("[environment] obstacle_manager succefully reset")
            else:
                rospy.loginfo("[environment] obstacle_manager not reset")
                self.is_initialized = False
                return self.is_initialized
        return self.is_initialized

    def monitor(self):
        if self.is_running:
            is_collision = self._check_collision()
            is_timeout = self._check_timeout()
            is_success = self._check_success()
            
            if is_collision or is_timeout or is_success:
                self.episode_stat["total"] += 1
                if is_collision:
                    self.episode_stat["collision"] += 1
                elif is_timeout:
                    self.episode_stat["timeout"] += 1
                else:
                    self.episode_stat["success"] += 1
                if self.auto_termination:
                    self.run_srv_callback(None)
                rospy.loginfo("\x1b[44m[environment] TOTAL %d EPISODES: SUCCESS(%d, %4.2f%%), COLLISION(%d, %4.2f%%), TIMEOUT(%d, %4.2f%%)\x1b[0m"
                              %(self.episode_stat["total"], self.episode_stat["success"], 100*float(self.episode_stat["success"])/self.episode_stat["total"],
                                self.episode_stat["collision"], 100*float(self.episode_stat["collision"])/self.episode_stat["total"],
                                self.episode_stat["timeout"], 100*float(self.episode_stat["timeout"])/self.episode_stat["total"]))
        else:
            if self.auto_start and self.is_initial_idle:
                self.run_srv_callback(None)
        
def main():
    rospy.init_node('environment_node')
    environment_node = Environment()

    rate = rospy.Rate(environment_node.freq)

    while not rospy.is_shutdown():
        if environment_node.is_initialized:
            environment_node.monitor()
        rate.sleep()

if __name__ == '__main__':
    main()