import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    import cvxpy as cp
    import numpy as np
    import plotly.graph_objects as go

    from dimp.robots import (
        OmniState, OmniInput, OmniRobot, RobotMPCData
    )
    return OmniInput, OmniRobot, OmniState, RobotMPCData, cp, go, mo, np


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Convex QCQP MPC
    
        **Scenario**: Omni-directional robot avoiding a circular obstacle and reaching a goal.
    
        The MPC problem is:
        $$\begin{align}
        &\underset{x}{\text{min}} \quad && \frac{1}{2} \sum_{k=1}^{n_c} (p_k - p_\mathrm{g})^2 \\
        &\text{s.t.} && s_{k+1} = A_k s_k + B_k u_k, \quad & k=1, \dots, n_c \qquad & \text{(Dynamics)} \\
        & && u_k^2 \leq u_{max}^2, \quad & k=1, \dots, n_c \qquad & \text{(Input limits)} \\
        & && (p_k - p_\mathrm{o})^T (\bar{p} - p_\mathrm{o}) \geq d_s, \quad & k=1, \dots, n_c \qquad & \text{(Collision avoidance)} \\
        \end{align}$$
        """
    )
    return


@app.cell
def _(OmniInput, OmniRobot, OmniState, RobotMPCData, cp, np):
    # ============================= Create The Data ============================= #

    ns = 2      # Number of states (x, y)
    ni = 2      # Number of inputs (vx, vy)

    nc = 2      # Number of control intervals

    s0 = cp.Parameter(ns)

    mpc_data = RobotMPCData(
        nc=nc,
        states_list=[OmniState(s0)] + [OmniState(cp.Variable(ns)) for _ in range(nc)],
        statesbar_list=[OmniState(s0)] + [OmniState(cp.Parameter(ns)) for _ in range(nc)],
        inputs_list=[OmniInput(cp.Variable(ni)) for _ in range(nc)],
        inputsbar_list=[OmniInput(cp.Parameter(ni)) for _ in range(nc)],
    )

    def populate_mpc_data(mpc_data):
        """Populate the MPC data with initial values."""
        mpc_data.statei[0].value = np.array([0.0, 0.0])
        for i in range(nc):
            mpc_data.statei[i+1].value = np.zeros(ns)
            mpc_data.statebari[i+1].value = np.zeros(ns)
            mpc_data.inputi[i].value = np.zeros(ni)
            mpc_data.inputbari[i].value = np.zeros(ni)

    populate_mpc_data(mpc_data)

    dt = cp.Parameter()
    dt.value = 0.1
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

    obstacle_position = np.array([5.0, 2.5])
    obstacle_radius = 1.0
    obstacle_constraint = [
        cp.transpose(mpc_data.statei[1] - obstacle_position) @ (mpc_data.statebari[1] - obstacle_position) - cp.pnorm(obstacle_radius, p=2) >= 0,
        cp.transpose(mpc_data.statei[2] - obstacle_position) @ (mpc_data.statebari[2] - obstacle_position) - cp.pnorm(obstacle_radius, p=2) >= 0,
    ]

    constraints = dynamics_constraints + input_constraints + obstacle_constraint

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    return (
        dt,
        mpc_data,
        ni,
        ns,
        obstacle_position,
        obstacle_radius,
        problem,
        s0,
    )


@app.cell
def _(mo):
    mo.md(r"### Simulate the Trajectory")
    return


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

        mpc_data.update_bar()

        states[i, :] = mpc_data.statei[1].value
        inputs[i, :] = mpc_data.inputi[1].value
    return states, steps


@app.cell
def _(mo):
    mo.md(r"### Plot the Trajectory")
    return


@app.cell
def _(go, obstacle_position, obstacle_radius, states, steps):
    xm, xM = states[:, 0].min() - 1, states[:, 0].max() + 1
    ym, yM = states[:, 1].min() - 1, states[:, 1].max() + 1

    fig = go.Figure(
        data=[
            go.Scatter(x=states[:, 0], y=states[:, 1],
                         mode="lines", name="Trajectory",
                         line=dict(width=2, color="rgba(0, 0, 255, 0.5)", dash='dot')),
            go.Scatter(x=[states[0, 0]], y=[states[0, 1]],
                         mode="markers", name="Robot",
                         marker=dict(color="blue", size=10)),
        ])

    fig.update_layout(width=600, height=450,
        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False, scaleanchor="y"),
        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
        title_text="Trajectory", title_x=0.5,
        updatemenus = [dict(type = "buttons",
            buttons = [
                dict(
                    args = [None, {"frame": {"duration": 10, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 10}, "mode": "immediate"}],
                    label = "Play",
                    method = "animate",
                )])],
        shapes=[
            dict(
                type="circle",
                name="Obstacle",
                showlegend=True,
                xref="x", yref="y",
                x0=obstacle_position[0] - obstacle_radius,
                y0=obstacle_position[1] - obstacle_radius,
                x1=obstacle_position[0] + obstacle_radius,
                y1=obstacle_position[1] + obstacle_radius,
                line=dict(color="red", width=2),
                fillcolor="rgba(255, 0, 0, 0.2)",
                layer="above",
            )
        ]
    )

    fig.update(frames=[
        go.Frame(
            data=[go.Scatter(x=[states[k, 0]], y=[states[k, 1]])],
            traces=[1]
        ) for k in range(steps)])

    fig.show()

    # fig.write_html("omni_robot_mpc.html", include_plotlyjs="cdn", full_html=False)
    return


if __name__ == "__main__":
    app.run()
