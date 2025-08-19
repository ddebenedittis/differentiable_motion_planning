import cvxpy as cp
import numpy as np

from dimp.robots.robot import GeneralRobot, GeneralRobotState, GeneralRobotInput


class UniState(GeneralRobotState):
    n = 3   # number of states
    
    property_names = ['state', 'x', 'y', 'theta']
        
    @property
    def x(self):
        return self.state[0:1]
    @x.setter
    def x(self, value):
        self.state[0:1] = value
    
    @property
    def y(self):
        return self.state[1:2]
    @y.setter
    def y(self, value):
        self.state[1:2] = value

    @property
    def theta(self):
        return self.state[2:3]
    @theta.setter
    def theta(self, value):
        self.state[2:3] = value

class UniInput(GeneralRobotInput):
    n = 2   # number of inputs
    
    property_names = ['input', 'v', 'omega']
    
    @property
    def v(self):
        return self.input[0:1]
    @v.setter
    def v(self, value):
        self.input[0:1] = value
    
    @property
    def omega(self):
        return self.input[1:2]
    @omega.setter
    def omega(self, value):
        self.input[1:2] = value


class UniRobot(GeneralRobot):
    def ct_dynamics(self, state, input, state_bar, input_bar):
        states_dot = cp.hstack([
            input[0] * np.cos(state_bar[2].value) - input_bar[0].value * np.sin(state_bar[2].value) * (state[2] - state_bar[2].value),
            input[0] * np.sin(state_bar[2].value) + input_bar[0].value * np.cos(state_bar[2].value) * (state[2] - state_bar[2].value),
            input[1],
        ])

        return states_dot
