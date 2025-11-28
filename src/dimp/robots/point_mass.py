import cvxpy as cp

from dimp.robots.robot import GeneralRobot, GeneralState, GeneralInput


class PointMassState(GeneralState):
    n = 4   # number of states
    
    property_names = ['state', 'x', 'y', 'vx', 'vy']
    
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
    def vx(self):
        return self.state[2:3]
    @vx.setter
    def vx(self, value):
        self.state[2:3] = value
    
    @property
    def vy(self):
        return self.state[3:4]
    @vy.setter
    def vy(self, value):
        self.state[3:4] = value

class PointMassInput(GeneralInput):
    n = 2   # number of inputs
    
    property_names = ['input', 'fx', 'fy']
    
    @property
    def fx(self):
        return self.input[0:1]
    @fx.setter
    def fx(self, value):
        self.input[0:1] = value
    
    @property
    def fy(self):
        return self.input[1:2]
    @fy.setter
    def fy(self, value):
        self.input[1:2] = value


class PointMassRobot(GeneralRobot):
    class PointMassParam:
        def __init__(self, mass=1.0):
            self.mass = mass
    
    p = PointMassParam()
    
    def ct_dynamics(self, state, input, state_bar=None, input_bar=None):
        states_dot = cp.hstack([
            state[2],
            state[3],
            input[0] / self.p.mass,
            input[1] / self.p.mass,
        ])
        
        return states_dot
