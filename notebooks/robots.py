import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from dimp.robots import OmniState, OmniInput, OmniRobot, RobotMPCData
    return OmniInput, OmniRobot, OmniState, RobotMPCData, np


@app.cell
def _(OmniInput, OmniRobot, OmniState, RobotMPCData, np):
    omni = OmniRobot()

    omni_state = OmniState()
    omni_input = OmniInput()
    omni_state.state = np.array([1.0, 2.0])
    omni_input.input = np.array([2.0, 3.0])

    data_mpc = RobotMPCData(4, [omni_state]*5, [omni_input]*4)
    return


if __name__ == "__main__":
    app.run()
