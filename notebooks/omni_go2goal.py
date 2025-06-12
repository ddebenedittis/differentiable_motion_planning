import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    import plotly.graph_objects as go
    return cp, go, np


@app.cell
def _(cp, np):
    ns = 2  # number of states
    ni = 2  # number of inputs

    s0 = cp.Parameter(2)           # initial state
    dt = cp.Parameter(nonneg=True)  # time step

    state = cp.Variable(ns)
    input = cp.Variable(ni)
    x = cp.hstack([state, input])

    # ============================= Define The Cost ============================= #

    p_goal = np.array([10, 5])

    A = np.block([
        [np.eye(ns), np.zeros((ns, ni))],
    ])
    b = - p_goal
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x + b, p=2))

    # ========================== Define The Constraints ========================= #

    v_max = 1.0

    constraints = [
        - state + input * dt + s0 == 0,
        cp.norm(input, p=2) - v_max <= 0,
    ]

    # ============================ Define The Problem =========================== #

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    return dt, input, ni, ns, problem, s0, state


@app.cell
def _(dt, input, ni, np, ns, problem, s0, state):
    steps = 200

    states = np.zeros((steps, ns))
    inputs = np.zeros((steps, ni))

    s0.value = np.array([0, 0])
    dt.value = 0.1

    for i in range(steps):

        problem.solve()

        s0.value = state.value

        states[i, :] = state.value
        inputs[i, :] = input.value
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
