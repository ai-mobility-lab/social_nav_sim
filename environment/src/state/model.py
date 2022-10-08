import numpy as np
from numpy.linalg import norm

class Model():
    def __init__(self, config):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.v_pref = config["v_pref"]
        self.radius = config["size"]
        self.px, self.py = config["init_pos"]
        self.theta = np.pi + np.arctan2(self.py, self.px)
        self.gx, self.gy = config["goal"]
        self.vx = 0.0
        self.vy = 0.0
        self.is_static = config["is_static"] if "is_static" in config.keys() else False
        self.T, self.T_inv = self.get_transformation()

    def get_position(self):
        return self.px, self.py

    def get_velocity(self):
        return self.vx, self.vy
    
    def get_goal_position(self):
        return self.gx, self.gy

    def get_rel_frame(self, px, py):
        p_rel = self.T_inv.dot(np.array([px, py, 1]))
        return p_rel[0], p_rel[1]

    def get_transformation(self):
        # T = [R|t]
        T = np.array([[np.cos(self.theta), -np.sin(self.theta), self.px],
                      [np.sin(self.theta),  np.cos(self.theta), self.py]])
        # inverse of T
        T_inv = np.linalg.inv(np.vstack([T, np.array([0, 0, 1])]))[:2,:]

        return T, T_inv

    def update(self, pos, vel, theta):
        """
        Perform an action and update the state
        """
        self.px = pos[0]
        self.py = pos[1]
        self.vx = vel[0]
        self.vy = vel[1]
        self.theta = theta
        self.T, self.T_inv = self.get_transformation()
        
    def is_goal_reached(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

