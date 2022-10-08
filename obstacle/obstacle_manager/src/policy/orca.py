import numpy as np
import rvo2

class ORCA():
    def __init__(self, config):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        self.config = config
        self.max_speed = 1 # the ego agent assumes that all other agents have this max speed
        self.sim = None
        self.time_step = config["time_step"]


    def predict(self, state):
        pass
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        self_state = state.self_state
        # max number of humans = current number of 'observable' humans
        max_neighbors = len(state.human_states)
        radius = state.self_state.radius
        # dynamic definition of params due to observability changes
        #params = self.config["neighbor_dist"], max_neighbors, self.config["time_horizon"], self.config["time_horizon_obst"]
        
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, self.config["neighbor_dist"], max_neighbors, self.config["time_horizon"], self.config["time_horizon_obst"], radius, self.max_speed)
            self.sim.addAgent(self_state.position, self.config["neighbor_dist"], max_neighbors, self.config["time_horizon"], self.config["time_horizon_obst"], self_state.radius + 0.01 + self.config["safety_space"],
                              self_state.v_pref, self_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, self.config["neighbor_dist"], max_neighbors, self.config["time_horizon"], self.config["time_horizon_obst"], human_state.radius + 0.01 + self.config["safety_space"],
                                  self.max_speed, human_state.velocity)
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        vx, vy = self.sim.getAgentVelocity(0)
        self.last_state = state

        return vx, vy
