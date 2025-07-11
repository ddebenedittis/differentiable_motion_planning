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
        # Classic QP
    
        The collision avoidance can be formulated as:
    
        - log barrier (not working for now)
        - quadratic penalty
        - others
    
        ## Convex QCQP MPC
    
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

    # ================================ Parameters =============================== #

    p_goal = np.array([10.0, 5.0])

    v_max = 1.0

    obstacle_position = np.array([5.0, 2.5])
    obstacle_radius = 1.0
    return (
        dt,
        mpc_data,
        nc,
        ni,
        ns,
        obstacle_position,
        obstacle_radius,
        p_goal,
        robot,
        s0,
        v_max,
    )


@app.cell
def _(cp, mpc_data, nc, np, p_goal, robot, v_max):
    # ============================ Create The Problem =========================== #

    def create_qcqp():
        objective = cp.Minimize(
            0.5 * cp.sum([cp.pnorm(mpc_data.statei[i+1] - p_goal) for i in range(nc)])
        )

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

        return problem

    qcqp_problem = create_qcqp()
    assert qcqp_problem.is_dpp()
    return (qcqp_problem,)


@app.cell
def _(mo):
    mo.md(r"### Simulate the Trajectory")
    return


@app.cell
def _(dt, ni, np, ns, s0):
    def simulate(problem, mpc_data):
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

        return states, inputs
    return (simulate,)


@app.cell
def _(mo):
    mo.md(r"### Plot the Trajectory")
    return


@app.cell
def _(go):
    def plot_trajectory(states, obstacle_position, obstacle_radius):
        steps = states.shape[0]

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

        fig.write_html("omni_robot_mpc.html", include_plotlyjs="cdn", full_html=False)
    return (plot_trajectory,)


@app.cell
def _(
    mpc_data,
    obstacle_position,
    obstacle_radius,
    plot_trajectory,
    qcqp_problem,
    simulate,
):
    def simulate_and_plot_qcqp():
        states, inputs = simulate(qcqp_problem, mpc_data)

        plot_trajectory(states, obstacle_position, obstacle_radius)

    simulate_and_plot_qcqp()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Differentiable QP
    
        ## Log Barrier
    
        Same scenario as above, but using a differentiable QP:
        $$\begin{align}
        &\underset{x}{\text{min}} \quad
          \frac{1}{2} \sum_{k=1}^{n_c}
          \bigl[\theta_1 (p_k - p_\mathrm{g})^2 - \theta_2 \log(|p - p_\mathrm{g}|_1 - d_\mathrm{s}) \bigr] \\
        &\text{s.t.} \quad
            \begin{alignedat}{3}
                s_{k+1} &= A_k s_k + B_k u_k & \qquad & k = 1, \dots, n_c & \qquad & \text{(Dynamics)} \\
                u_k^2 &\le u_{\max}^2 & & k = 1, \dots, n_c & & \text{(Input limits)} \\
            \end{alignedat}
        \end{align}$$
    
        No bueno :-1: for CVXPY, so we rewrite it as:
        $$\begin{align}
        &\underset{x}{\text{min}} \quad
          \frac{1}{2} \sum_{k=1}^{n_c}
          \bigl[\theta_1 (p_k - p_\mathrm{g})^2 - \theta_2 \log(|aux_k|_1 - d_\mathrm{s}) + k |aux_k|_2^2\bigr] \\
        &\text{s.t.} \quad
            \begin{alignedat}{3}
                s_{k+1} &= A_k s_k + B_k u_k & \qquad & k = 1, \dots, n_c & \qquad & \text{(Dynamics)} \\
                u_k^2 &\le u_{\max}^2 & & k = 1, \dots, n_c & & \text{(Input limits)} \\
                aux_k &\ge 0 & & k = 1, \dots, n_c & & \text{(Auxiliary)} \\
                - aux_k &\le p - p_\mathrm{o} \le aux_k & & k = 1, \dots, n_c & & \text{(Auxiliary)}
            \end{alignedat}
        \end{align}$$
        """
    )
    return


@app.cell
def _(
    cp,
    mpc_data,
    nc,
    obstacle_position,
    obstacle_radius,
    p_goal,
    robot,
    v_max,
):
    aux1 = cp.Variable(nc*2, nonneg=True)

    def create_dqp_log_barrier(theta):
        objective = cp.Minimize(
            + 0.5 * theta[0] * cp.sum([cp.pnorm(mpc_data.statei[i+1] - p_goal) for i in range(nc)]) \
            - 0.5 * theta[1] * cp.sum([cp.log(aux1[2*i] + aux1[2*i+1] - 2*obstacle_radius) for i in range(nc)])
            + 1e6 * cp.pnorm(aux1, p=2)
        )

        dynamics_constraints = robot.dt_dynamics_constraint()

        input_constraints = [
            cp.norm(mpc_data.inputi[0], p=2) - v_max <= 0,
            cp.norm(mpc_data.inputi[1], p=2) - v_max <= 0,
        ]

        aux1_constraints = [
            aux1[i] >= 0 for i in range(nc*2)
        ] + [
            aux1[2*i] >= mpc_data.xi[i+1] - obstacle_position[0]
            for i in range(nc)
        ] + [
            - aux1[2*i] <= mpc_data.xi[i+1] - obstacle_position[0]
            for i in range(nc)
        ] + [
            aux1[2*i+1] >= mpc_data.yi[i+1] - obstacle_position[1]
            for i in range(nc)
        ] + [
            - aux1[2*i+1] <= mpc_data.yi[i+1] - obstacle_position[1]
            for i in range(nc)
        ]

        constraints = dynamics_constraints + input_constraints + aux1_constraints

        problem = cp.Problem(objective, constraints)

        assert problem.is_dpp(), "The problem is not DPP"

        return problem
    return (create_dqp_log_barrier,)


@app.cell
def _(
    cp,
    create_dqp_log_barrier,
    mpc_data,
    np,
    obstacle_position,
    obstacle_radius,
    plot_trajectory,
    simulate,
):
    def simulate_and_plot_dqp_log_barrier():
        theta = cp.Parameter(2, nonneg=True)

        theta.value = np.array([1e0, 1e0])

        dqp = create_dqp_log_barrier(theta)

        states, inputs = simulate(dqp, mpc_data)

        plot_trajectory(states, obstacle_position, obstacle_radius)

    simulate_and_plot_dqp_log_barrier()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Quadratic Penalty
    
        $$\begin{align}
        &\underset{x}{\text{min}} \quad
          \frac{1}{2} \sum_{k=1}^{n_c}
          \bigl[\theta_1 |p_k - p_\mathrm{g}|_2^2 - \theta_2 s_k^2 \bigr] \\
        &\text{s.t.} \quad
            \begin{alignedat}{3}
                s_{k+1} &= A_k s_k + B_k u_k & \qquad & k = 1, \dots, n_c & \qquad & \text{(Dynamics)} \\
                u_k^2 &\le u_{\max}^2 & & k = 1, \dots, n_c & & \text{(Input limits)} \\
                (p_k - p_\mathrm{o})^T (\bar{p} - p_\mathrm{o}) &\ge d_s^2 - s_k & & k = 1, \dots, n_c & & \text{(Collision avoidance)} \\
                s_k & \geq 0 & & k = 1, \dots, n_c & & \text{(Auxiliary)}
            \end{alignedat}
        \end{align}$$
        """
    )
    return


@app.cell
def _(
    cp,
    mpc_data,
    nc,
    obstacle_position,
    obstacle_radius,
    p_goal,
    robot,
    v_max,
):
    aux2 = cp.Variable(nc*2, nonneg=True)

    def create_dqp_quadratic_penalty(theta):
        objective = cp.Minimize(
            + 0.5 * theta[0] * cp.sum([cp.pnorm(mpc_data.statei[i+1] - p_goal) for i in range(nc)]) \
            + theta[1] * cp.sum(cp.pnorm(aux2, p=2))
        )

        dynamics_constraints = robot.dt_dynamics_constraint()

        input_constraints = [
            cp.norm(mpc_data.inputi[0], p=2) - v_max <= 0,
            cp.norm(mpc_data.inputi[1], p=2) - v_max <= 0,
        ]

        collision_constraints = [
            cp.transpose(mpc_data.statei[i+1] - obstacle_position) @ (mpc_data.statebari[i+1] - obstacle_position) - cp.pnorm(obstacle_radius, p=2) >= - aux2[i]
            for i in range(nc)
        ] + [
            aux2[i] >= 0 for i in range(nc*2)
        ]

        constraints = dynamics_constraints + input_constraints + collision_constraints

        problem = cp.Problem(objective, constraints)

        assert problem.is_dpp(), "The problem is not DPP"

        return problem

    return (create_dqp_quadratic_penalty,)


@app.cell
def _(
    cp,
    create_dqp_quadratic_penalty,
    mpc_data,
    np,
    obstacle_position,
    obstacle_radius,
    plot_trajectory,
    simulate,
):
    def simulate_and_plot_dqp_quadratic_penalty():
        theta = cp.Parameter(2, nonneg=True)

        theta.value = np.array([1e0, 1e0])

        dqp = create_dqp_quadratic_penalty(theta)

        states, inputs = simulate(dqp, mpc_data)

        plot_trajectory(states, obstacle_position, obstacle_radius)

    simulate_and_plot_dqp_quadratic_penalty()
    return


if __name__ == "__main__":
    app.run()
