from dimp.robots.robot import GeneralInput


class ForceInput(GeneralInput):
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
