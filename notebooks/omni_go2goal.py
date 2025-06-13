import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    import plotly.graph_objects as go

    from dimp.robots import (
        OmniState, OmniInput, OmniRobot, RobotMPCData
    )
    return OmniInput, OmniRobot, OmniState, RobotMPCData, cp, go, np


@app.cell
def _(OmniInput, OmniRobot, OmniState, RobotMPCData, cp, np):
    ns = 2      # Number of states (x, y)
    ni = 2      # Number of inputs (vx, vy)

    nc = 2      # Number of control intervals

    s0 = cp.Parameter(ns)

    mpc_data = RobotMPCData(
        nc=nc,
        states_list=[OmniState(s0)] + [OmniState(cp.Variable(ns)) for _ in range(nc)],
        inputs_list=[OmniInput(cp.Variable(ni)) for _ in range(nc)],
    )

    dt = cp.Parameter()
    robot = OmniRobot(dt=dt, mpc_data=mpc_data)

    # ============================= Define The Cost ============================= #

    p_goal = np.array([10.0, 5.0])

    objective = cp.Minimize(
        0.5 * cp.sum([cp.pnorm(mpc_data.statei[i+1] - p_goal) for i in range(nc)])
    )

    # ========================== Define The Constraints ========================= #

    v_max = 1.0

    dynamics_constraints = robot.dt_dynamics_constraint()

    input_constraints = [
        cp.norm(mpc_data.inputi[0], p=2) - v_max <= 0,
        cp.norm(mpc_data.inputi[1], p=2) - v_max <= 0,
    ]

    constraints = dynamics_constraints + input_constraints

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    return dt, mpc_data, ni, ns, problem, s0


@app.cell
def _(dt, mpc_data, ni, np, ns, problem, s0):
    steps = 200

    states = np.zeros((steps, ns))
    inputs = np.zeros((steps, ni))

    s0.value = np.array([0, 0])
    dt.value = 0.1

    for i in range(steps):

        problem.solve()

        s0.value = mpc_data.statei[1].value

        states[i, :] = mpc_data.statei[1].value
        inputs[i, :] = mpc_data.inputi[1].value
    return states, steps


@app.cell
def _(go, states, steps):
    xm, xM = states[:, 0].min() - 1, states[:, 0].max() + 1
    ym, yM = states[:, 1].min() - 1, states[:, 1].max() + 1

    fig = go.Figure(
        data=[
            go.Scatter(x=states[:, 0], y=states[:, 1],
                         mode="lines",
                         line=dict(width=2, color="blue")),
            go.Scatter(x=[states[0, 0]], y=[states[0, 1]],
                         mode="markers",
                         marker=dict(color="red", size=10))
        ])

    fig.update_layout(width=600, height=450,
        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
        title_text="Trajectory", title_x=0.5,
        updatemenus = [dict(type = "buttons",
            buttons = [
                dict(
                    args = [None, {"frame": {"duration": 10, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 10}}],
                    label = "Play",
                    method = "animate",
                )])]
    )

    fig.update(frames=[
        go.Frame(
            data=[go.Scatter(x=[states[k, 0]], y=[states[k, 1]])],
            traces=[1]
        ) for k in range(steps)])

    fig.show()
    return


if __name__ == "__main__":
    app.run()
