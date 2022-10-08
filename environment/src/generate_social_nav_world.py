#!/usr/bin/env python
# coding=utf-8

'''
Author:Tai Lei
Date:
Info:
'''

import random
import numpy as np
from lxml import etree
from lxml.etree import Element
from utils import *

import rospkg
import rospy

#####################################################################################
## LOAD ROSPARAM ##
#####################################################################################
# get path to the plugin
rospack = rospkg.RosPack()
plugin_pkg_path = rospack.get_path("animated_obstacle_plugin")
plugin_path = plugin_pkg_path + "/lib/libanimated_obstacle_plugin_ros.so"

# get path to empty.world file
manager_pkg_path = rospack.get_path("environment")
tree_ = etree.parse(manager_pkg_path+'/worlds/empty.world')
world_ = tree_.getroot().getchildren()[0]

# get ros parameters
## total number of obstacles
num_obs = rospy.get_param("/obstacle/num_obstacle", 5)
## load skin list for human actors
actor_default_height = rospy.get_param("/obstacle/visual/actor_default_height")
#####################################################################################

#####################################################################################
## INITIALIZE AEGNT & OBSTACLES ##
#####################################################################################
agent_info, obs_info = initialize_model_locations()

#####################################################################################
## SET ROSPARAM ##
#####################################################################################
# set rosparam for publishing agent information
rospy.set_param("agent_information", agent_info)

# set rosparam for publishing obstacle information
rospy.set_param("obstacle_information", obs_info)
#####################################################################################

#####################################################################################
## CREATE WORLD FILE ##
#####################################################################################
# enter models of obstacles into world file
for i in range(num_obs):
    obs_dict = obs_info[i]
    if not obs_dict["is_human"]:
        obstacle_info_stack = Element("include")
        
        # name
        name = Element("name")
        name.text = obs_dict["name"]
        obstacle_info_stack.append(name)
        
        # pose
        pose = Element("pose")
        ## x y z roll pitch yaw
        pose.text = "%f %f 0 0 0 0"%(obs_dict["init_pos"][0], obs_dict["init_pos"][1])
        obstacle_info_stack.append(pose)
        
        # path to the model
        model = Element("uri")
        model.text = obs_dict["model_path"]
        obstacle_info_stack.append(model)
        
    else:
        # compute size proportion
        scale = 1 + np.clip(obs_dict["size"] - 0.3, -0.2, 0.2)
        obstacle_info_stack = Element("actor", name=obs_dict["name"])

        pose = Element("pose")
        ## x y z roll pitch yaw
        pose.text = "%f %f %f 0 0 0"%(obs_dict["init_pos"][0], obs_dict["init_pos"][1], actor_default_height*scale)
        obstacle_info_stack.append(pose)
        
        skin = Element("skin")
        skin_fn = Element("filename")
        skin_fn.text = obs_dict["model_path"]
        skin_scale = Element("scale")
        skin_scale.text = "%f"%scale
        skin.append(skin_fn)
        skin.append(skin_scale)
        obstacle_info_stack.append(skin)

        animation = Element("animation", name="walking")
        animate_fn = Element("filename")
        if obs_dict["is_static"]:
            animate_fn.text = "stand.dae"
        else:
            animate_fn.text = "walk.dae"
        animate_scale = Element("scale")
        animate_scale.text = "1"
        interpolate_x = Element("interpolate_x")
        interpolate_x.text = "true"
        animation.append(animate_fn)
        animation.append(animate_scale)
        animation.append(interpolate_x)
        obstacle_info_stack.append(animation)

        plugin = Element("plugin", name="control", filename=plugin_path)
        ignore_obstacle = Element("ignore_obstacles")
        model_ground_plane = Element("model")
        model_ground_plane.text = "ground_plane"
        ignore_obstacle.append(model_ground_plane)
        is_static = Element("is_static")
        is_static.text = str(obs_dict["is_static"]).lower()
        plugin.append(ignore_obstacle)
        plugin.append(is_static)
        obstacle_info_stack.append(plugin)

    world_.append(obstacle_info_stack)
    
# add agent's goal
goal_model = Element("include")
        
# name
name = Element("name")
name.text = "goal"
goal_model.append(name)

# pose
pose = Element("pose")
## x y z roll pitch yaw
pose.text = "%f %f 0 0 0 0"%(agent_info["goal"][0], agent_info["goal"][1])
goal_model.append(pose)

# path to the model
model = Element("uri")
model.text = "model://goal_marker"
goal_model.append(model)

world_.append(goal_model)


tree_.write(manager_pkg_path+'/worlds/social_nav_env.world', pretty_print=True, xml_declaration=True, encoding="utf-8")
#####################################################################################