import numpy as np

class SOCIAL_FORCE():
    def __init__(self, config):
        self.config = config
        self.time_step = self.config["time_step"]
        self.A = self.config["A"]
        self.B = self.config["B"]
        self.KI = self.config["KI"]

    def predict(self, state):
        """
        Produce action for agent with circular specification of social force model.
        """
        # Pull force to goal
        delta_x = state.self_state.gx - state.self_state.px
        delta_y = state.self_state.gy - state.self_state.py
        dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
        desired_vx = (delta_x / dist_to_goal) * state.self_state.v_pref
        desired_vy = (delta_y / dist_to_goal) * state.self_state.v_pref
        curr_delta_vx = self.KI * (desired_vx - state.self_state.vx)
        curr_delta_vy = self.KI * (desired_vy - state.self_state.vy)
        
        # Push force(s) from other agents
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in state.human_states:
            delta_x = state.self_state.px - other_human_state.px
            delta_y = state.self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)
            interaction_vx += self.A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / self.B) * (delta_x / dist_to_human)
            interaction_vy += self.A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / self.B) * (delta_y / dist_to_human)

        # Sum of push & pull forces
        total_delta_vx = (curr_delta_vx + interaction_vx) * self.time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * self.time_step

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        new_vx = state.self_state.vx + total_delta_vx
        new_vy = state.self_state.vy + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])

        if act_norm > state.self_state.v_pref:
            new_vx = new_vx / act_norm * state.self_state.v_pref
            new_vy = new_vy / act_norm * state.self_state.v_pref
            
        return new_vx, new_vy