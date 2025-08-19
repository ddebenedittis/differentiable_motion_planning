from dimp.robots.robot import GeneralRobot, GeneralRobotState, GeneralRobotInput


class OmniState(GeneralRobotState):
    n = 2   # number of states
    
    property_names = ['state', 'x', 'y']
        
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
    
class OmniInput(GeneralRobotInput):
    n = 2   # number of inputs
    
    property_names = ['input', 'vx', 'vy']
    
    @property
    def vx(self):
        return self.input[0:1]
    
    @vx.setter
    def vx(self, value):
        self.input[0:1] = value
    
    @property
    def vy(self):
        return self.input[1:2]
    
    @vy.setter
    def vy(self, value):
        self.input[1:2] = value


class OmniRobot(GeneralRobot):
    def ct_dynamics(self, state, input, state_bar=None, input_bar=None):
        states_dot = input
        
        return states_dot
