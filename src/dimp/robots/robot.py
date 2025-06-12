from abc import ABC, abstractmethod
from copy import deepcopy

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
        assert len(self._state) == self.n, f"'{self.property_names[0]} must have length {self.n}."
        
        return self._state
    @state.setter
    def state(self, value):
        assert value is not None, f"'{self.property_names[0]} cannot be None."
        assert len(value) == self.n, f"'{self.property_names[0]} must have length {self.n}."
        
        self._state = deepcopy(value)
        
class GeneralRobotInput(GeneralRobotData):
    n = 0      # number of input
    
    property_names = ["input"]
    
    def __init__(self, input = None):
        self._input = input
    
    @property
    def input(self):
        assert self._input is not None, f"'{self.property_names[0]} has not been initialized."
        assert len(self._input) == self.n, f"'{self.property_names[0]} must have length {self.n}."
        
        return self._input
    @input.setter
    def input(self, value):
        assert value is not None, f"'{self.property_names[0]} cannot be None."
        assert len(value) == self.n, f"'{self.property_names[0]} must have length {self.n}."
        
        self._input = deepcopy(value)


class RobotMPCData:
    class _AttrIndexer:
        def __init__(self, datas, attr):
            self._datas = datas
            self._attr = attr
        def __getitem__(self, i):
            return getattr(self._datas[i], self._attr)
        def __setitem__(self, i, value):
            setattr(self._datas[i], self._attr, value)

    def __init__(self, nc: int, states_list: list, inputs_list: list):
        assert isinstance(nc, int) and nc > 0, "Number of control steps (nc) must be a positive integer."
        super().__setattr__('nc', nc)
        
        assert isinstance(states_list, list) and all(isinstance(d, GeneralRobotData) for d in states_list), \
            "states_list must be a list of GeneralRobotData instances."
        super().__setattr__('states_list', [deepcopy(d) for d in states_list])
        
        assert isinstance(inputs_list, list) and all(isinstance(d, GeneralRobotData) for d in inputs_list), \
            "inputs_list must be a list of GeneralRobotData instances."
        super().__setattr__('inputs_list', [deepcopy(d) for d in inputs_list])
        
        super().__setattr__('_states_names', set(states_list[0].property_names) if states_list else set())
        super().__setattr__('_inputs_names', set(inputs_list[0].property_names) if inputs_list else set())

    def __getattr__(self, name):
        for names, datas_list in zip([self._states_names, self._inputs_names], [self.states_list, self.inputs_list]):
            if name in self._states_names:
                return np.concatenate([getattr(d, name) for d in datas_list])
            if name.endswith('i') and name[:-1] in names:
                return RobotMPCData._AttrIndexer(datas_list, name[:-1])
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in ('nc', 'names_list', 'inputs_list'):
            super().__setattr__(name, value)
            return
            
        for names, datas_list in zip([self._states_names, self._inputs_names], [self.states_list, self.inputs_list]):
            if name in names:
                if len(value) != len(datas_list):
                    raise ValueError(f"Expected {len(datas_list)} values, got {len(value)}")
                for d, v in zip(datas_list, value):
                    setattr(d, name, deepcopy(v))
            else:
                super().__setattr__(name, deepcopy(value))      


class GeneralRobot(ABC):
    def __init__(self, dt: float = 0.1):
        assert dt > 0, "Time step must be positive."
        self.dt = dt
        
    @abstractmethod
    def ct_dynamics(self, state, input):
        pass
    
    def dt_dynamics(self, state, input):
        return self.ct_dynamics(state, input) * self.dt
