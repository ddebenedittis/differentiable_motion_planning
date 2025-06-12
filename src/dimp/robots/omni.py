from dimp.robots.robot import GeneralRobot, GeneralRobotState, GeneralRobotInput


class OmniState(GeneralRobotState):
    n = 2   # number of states
    
    property_names = ['state', 'x', 'y']
        
    @property
    def x(self):
        return self.state[0:1]
    
    @property
    def y(self):
        return self.state[1:2]
    
class OmniInput(GeneralRobotInput):
    n = 2   # number of inputs
    
    property_names = ['input', 'vx', 'vy']
    
    @property
    def vx(self):
        return self.input[0:1]
    
    @property
    def vy(self):
        return self.input[1:2]


class OmniRobot(GeneralRobot):
    def ct_dynamics(self, state: OmniState, input: OmniInput) -> OmniState:
        states_dot = input.input
        
        return OmniState(states_dot)
