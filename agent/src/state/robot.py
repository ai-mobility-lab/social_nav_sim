import numpy as np
from numpy.linalg import norm
# import packages in the view of src directory
from state.state import ObservableState, FullState, JointState
from policy.orca import ORCA
from policy.social_force import SOCIAL_FORCE
from policy.dummy import Dummy
from policy.srnn import SRNN

class Robot(object):
    def __init__(self, config):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.v_pref = config["v_pref"]
        self.radius = config["size"]
        if config["policy"].lower() == "srnn":
            self.policy = SRNN(config["policy_config"])
        elif config["policy"].lower() == "orca":
            self.policy = ORCA(config["policy_config"])
        elif config["policy"].lower() == "social_force":
            self.policy = SOCIAL_FORCE(config["policy_config"])
        elif config["policy"].lower() == "dummy":
            self.policy = Dummy()
        self.FOV = np.pi * config["FOV"]
        self.px, self.py = config["init_pos"]
        self.theta = np.pi + np.arctan2(self.py, self.px)
        self.gx, self.gy = config["goal"]
        self.vx = 0.0
        self.vy = 0.0
        self.vx_b = 0.0
        self.vy_b = 0.0
        self.last_action = [0.0, 0.0]
        self.last_pos = config["init_pos"]
        self.last_theta = self.theta
        self.time_step = config["time_step"]
        self.policy.time_step = config["time_step"]
        self.R, self.T, self.T_inv = self.get_transformation()
        self.kinematics = config["kinematics"]
        
    def set_last_action(self, action):
        self.last_action = action
        if self.kinematics == "unicycle":
            self.last_theta = (self.theta + action[1]) % (2*np.pi)
        else:
            self.last_theta = np.arctan2(action[1], action[0])
            self.last_pos = [self.px + action[0]*self.time_step, self.py + action[1]*self.time_step]

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_observable_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius]

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_full_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta]

    def get_full_state_list_noV(self):
        return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref, self.last_theta]

    def get_full_relative_state_list_noV(self):
        if self.kinematics == "unicycle":
            gx_rel, gy_rel = self.get_rel_frame(self.gx, self.gy)
            return [self.radius, gx_rel, gy_rel, self.v_pref]
        else:
            return [self.radius, self.gx - self.px, self.gy - self.py, self.v_pref]

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def get_rel_frame(self, px, py):
        p_rel = self.T_inv.dot(np.array([px, py, 1]))
        return p_rel[0], p_rel[1]

    def get_rel_pos(self, px, py):
        if self.kinematics == "unicycle":
            pos_rel = self.get_rel_frame(px, py)
        else:
            pos_rel = [px - self.px, py - self.py]
        return np.array(pos_rel)

    def get_vel_in_body(self, vx, vy):
        vel = self.R.T.dot(np.array([vx, vy]))
        return vel[0], vel[1]

    def get_transformation(self):
        # T = [R|t]
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                      [np.sin(self.theta),  np.cos(self.theta)]])
        T = np.hstack([R, np.array([self.px, self.py])[:,None]])
        # inverse of T
        T_inv = np.linalg.inv(np.vstack([T, np.array([0, 0, 1])]))[:2,:]

        return R, T, T_inv

    def act(self, ob=None):
        """
        Determine action values based on observation
        :param ob:
        :return:
        """

        if isinstance(self.policy, SRNN):
            state = ob
        elif isinstance(self.policy, Dummy):
            state = None
        else:
            state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def update(self, pos, vel, theta):
        """
        Perform an action and update the state
        """
        self.px = pos[0]
        self.py = pos[1]
        self.vx = vel[0]
        self.vy = vel[1]
        self.theta = theta
        self.R, self.T, self.T_inv = self.get_transformation()
        self.vx_b, self.vy_b = self.get_vel_in_body(self.vx, self.vy)

    def is_goal_reached(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

