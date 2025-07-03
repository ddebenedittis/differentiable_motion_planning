from abc import ABC, abstractmethod

import numpy as np


class GeneralRobotData:
    n = 0      # number of state
    
    property_names = ["data"]
    
class GeneralRobotState(GeneralRobotData):
    n = 0      # number of state
    
    property_names = ["state"]
    
    def __init__(self, state = None):
        self._state = state
    
    @property
    def state(self):
        assert self._state is not None, f"'{self.property_names[0]} has not been initialized."
        assert self._state.shape == (self.n,), f"'{self.property_names[0]} must have length {self.n}."
        
        return self._state
    
    @state.setter
    def state(self, value):
        assert value is not None, f"'{self.property_names[0]} cannot be None."
        assert value.shape == (self.n,), f"'{self.property_names[0]} must have length {self.n}."
        
        self._state = value
        
class GeneralRobotInput(GeneralRobotData):
    n = 0      # number of input
    
    property_names = ["input"]
    
    def __init__(self, input = None):
        self._input = input
    
    @property
    def input(self):
        assert self._input is not None, f"'{self.property_names[0]} has not been initialized."
        assert self._input.shape == (self.n,), f"'{self.property_names[0]} must have length {self.n}."
        
        return self._input
    
    @input.setter
    def input(self, value):
        assert value is not None, f"'{self.property_names[0]} cannot be None."
        assert value.shape == (self.n,), f"'{self.property_names[0]} must have length {self.n}."
        
        self._input = value


class RobotMPCData:
    class _AttrIndexer:
        def __init__(self, datas, attr):
            self._datas = datas
            self._attr = attr
        def __getitem__(self, i):
            return getattr(self._datas[i], self._attr)
        def __setitem__(self, i, value):
            setattr(self._datas[i], self._attr, value)

    def __init__(
        self,
        nc: int,
        states_list: list | None = None,
        inputs_list: list | None = None,
        statesbar_list: list | None = None,
        inputsbar_list: list | None = None,
    ):
        assert isinstance(nc, int) and nc > 0, "Number of control steps (nc) must be a positive integer."
        super().__setattr__('nc', nc)
        
        assert isinstance(states_list, list) and all(isinstance(d, GeneralRobotData) for d in states_list), \
            "states_list must be a list of GeneralRobotData instances."
        super().__setattr__('states_list', [d for d in states_list])
        
        assert isinstance(statesbar_list, list) and all(isinstance(d, GeneralRobotData) for d in statesbar_list), \
            "statesbar_list must be a list of GeneralRobotData instances."
        super().__setattr__('statesbar_list', [d for d in statesbar_list])
        
        assert isinstance(inputs_list, list) and all(isinstance(d, GeneralRobotData) for d in inputs_list), \
            "inputs_list must be a list of GeneralRobotData instances."
        super().__setattr__('inputs_list', [d for d in inputs_list])
        
        assert isinstance(inputsbar_list, list) and all(isinstance(d, GeneralRobotData) for d in inputsbar_list), \
            "inputsbar_list must be a list of GeneralRobotData instances."
        super().__setattr__('inputsbar_list', [d for d in inputsbar_list])
        
        super().__setattr__('_states_names', set(states_list[0].property_names) if states_list else set())
        super().__setattr__('_inputs_names', set(inputs_list[0].property_names) if inputs_list else set())

    def __getattr__(self, name):
        if name == 'nc':
            return self.nc
        
        if 'bar' not in name:
            lists = [self.states_list, self.inputs_list]
        else:
            lists = [self.statesbar_list, self.inputsbar_list]
            name = name.replace('bar', '')
            
        for names, datas_list in zip([self._states_names, self._inputs_names], lists):
            if name in self._states_names:
                return np.concatenate([getattr(d, name) for d in datas_list])
            if name.endswith('i') and name[:-1] in names:
                return RobotMPCData._AttrIndexer(datas_list, name[:-1])
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in ('nc', 'names_list', 'inputs_list'):
            super().__setattr__(name, value)
            return
        
        if 'bar' not in name:
            lists = [self.states_list, self.inputs_list]
        else:
            lists = [self.statesbar_list, self.inputsbar_list]
            name = name.replace('bar', '')
            
        for names, datas_list in zip([self._states_names, self._inputs_names], lists):
            if name in names:
                if len(value) != len(datas_list):
                    raise ValueError(f"Expected {len(datas_list)} values, got {len(value)}")
                for d, v in zip(datas_list, value):
                    setattr(d, name, v)
            else:
                super().__setattr__(name, value)
                
    def update_bar(self):
        self.statebari[0].value = self.statei[0].value
        for i in range(self.nc):
            self.statebari[i+1].value = self.statei[i+1].value
            self.inputbari[i].value = self.inputi[i].value


class GeneralRobot(ABC):
    def __init__(self, dt: float = 0.1, mpc_data: RobotMPCData | None = None):
        self.dt = dt
        
        self.mpc_data: RobotMPCData | None = mpc_data
        
    @abstractmethod
    def ct_dynamics(self, state, input):
        pass
    
    def dt_dynamics(self, state, input):
        return self.ct_dynamics(state, input) * self.dt

    def dt_dynamics_constraint(self):
        return [
            - self.mpc_data.statei[i+1] \
                + self.dt_dynamics(self.mpc_data.statei[i], self.mpc_data.inputi[i]) \
                + self.mpc_data.statei[i] == 0 \
            for i in range(self.mpc_data.nc)
        ]
