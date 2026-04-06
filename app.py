from __future__ import annotations

from io import StringIO
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "Loss Surface Visualizer"
M_RANGE = (-5.0, 5.0)
B_RANGE = (-5.0, 5.0)
GRID_SIZE = 100
DEFAULT_POINTS = 25
DEFAULT_NOISE = 1.2
DEFAULT_DATA_SEED = 11
DEFAULT_M = 0.0
DEFAULT_B = 0.0
DEFAULT_LR = 0.05
DEFAULT_STEPS = 22
DEFAULT_GD_SEED = 9


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_state() -> None:
    """Populate session state with default values used by the control panel."""
    defaults = {
        "m": DEFAULT_M,
        "b": DEFAULT_B,
        "learning_rate": DEFAULT_LR,
        "gd_steps": DEFAULT_STEPS,
        "show_gd": True,
        "animate_gd": False,
        "data_seed": DEFAULT_DATA_SEED,
        "gd_seed": DEFAULT_GD_SEED,
        "noise_scale": DEFAULT_NOISE,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_controls() -> None:
    """Reset the interactive controls to their default values."""
    st.session_state["m"] = DEFAULT_M
    st.session_state["b"] = DEFAULT_B
    st.session_state["learning_rate"] = DEFAULT_LR
    st.session_state["gd_steps"] = DEFAULT_STEPS
    st.session_state["show_gd"] = True
    st.session_state["animate_gd"] = False
    st.session_state["gd_seed"] = DEFAULT_GD_SEED


@st.cache_data(show_spinner=False)
def generate_data(
    num_points: int = DEFAULT_POINTS,
    noise_scale: float = DEFAULT_NOISE,
    seed: int = DEFAULT_DATA_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small noisy linear dataset for exploration."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-4.5, 4.5, num_points)
    true_m = 2.1
    true_b = 0.8
    noise = rng.normal(0.0, noise_scale, size=num_points)
    y = true_m * x + true_b + noise
    return x, y


def compute_loss(x: np.ndarray, y: np.ndarray, m: float, b: float) -> float:
    """Compute mean squared error for the line y = m*x + b."""
    predictions = m * x + b
    return float(np.mean((y - predictions) ** 2))


@st.cache_data(show_spinner=False)
def compute_loss_surface(
    x: np.ndarray,
    y: np.ndarray,
    m_range: tuple[float, float] = M_RANGE,
    b_range: tuple[float, float] = B_RANGE,
    resolution: int = GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the loss surface across a grid of slope and intercept values."""
    m_values = np.linspace(m_range[0], m_range[1], resolution)
    b_values = np.linspace(b_range[0], b_range[1], resolution)
    m_grid, b_grid = np.meshgrid(m_values, b_values)

    predictions = m_grid[..., None] * x + b_grid[..., None]
    losses = np.mean((y - predictions) ** 2, axis=2)
    return m_grid, b_grid, losses


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    start_m: float,
    start_b: float,
    learning_rate: float,
    steps: int,
) -> list[tuple[float, float, float]]:
    """Simulate gradient descent and return the parameter path."""
    m = float(start_m)
    b = float(start_b)
    path: list[tuple[float, float, float]] = []
    n = len(x)

    for _ in range(steps):
        predictions = m * x + b
        error = predictions - y
        loss = float(np.mean(error**2))
        path.append((m, b, loss))

        grad_m = (2.0 / n) * np.sum(error * x)
        grad_b = (2.0 / n) * np.sum(error)

        m -= learning_rate * grad_m
        b -= learning_rate * grad_b

    path.append((m, b, compute_loss(x, y, m, b)))
    return path


def get_global_minimum(
    m_grid: np.ndarray, b_grid: np.ndarray, losses: np.ndarray
) -> tuple[float, float, float]:
    """Return the lowest-loss parameter pair found on the surface grid."""
    min_index = np.unravel_index(np.argmin(losses), losses.shape)
    return (
        float(m_grid[min_index]),
        float(b_grid[min_index]),
        float(losses[min_index]),
    )


def dataset_to_csv(x: np.ndarray, y: np.ndarray) -> str:
    """Convert the synthetic dataset to CSV text for download."""
    buffer = StringIO()
    buffer.write("x,y\n")
    for x_value, y_value in zip(x, y):
        buffer.write(f"{x_value:.6f},{y_value:.6f}\n")
    return buffer.getvalue()


def plot_dataset(x: np.ndarray, y: np.ndarray, m: float, b: float) -> go.Figure:
    """Plot the noisy dataset and the current regression line."""
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    y_line = m * x_line + b

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Synthetic data",
            marker=dict(size=10, color="#1d4ed8", line=dict(width=1, color="#ffffff")),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra>Point</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Current model",
            line=dict(color="#dc2626", width=3),
            hovertemplate="x=%{x:.2f}<br>prediction=%{y:.2f}<extra>Line</extra>",
        )
    )
    fig.update_layout(
        title="Dataset and Current Regression Line",
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_surface(
    m_grid: np.ndarray,
    b_grid: np.ndarray,
    losses: np.ndarray,
    current_point: tuple[float, float, float],
    global_minimum: tuple[float, float, float],
) -> go.Figure:
    """Render an interactive 3D surface plot of the loss landscape."""
    current_m, current_b, current_loss = current_point
    min_m, min_b, min_loss = global_minimum

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=m_grid,
            y=b_grid,
            z=losses,
            colorscale="Viridis",
            opacity=0.95,
            showscale=True,
            colorbar=dict(title="MSE"),
            hovertemplate="m=%{x:.2f}<br>b=%{y:.2f}<br>loss=%{z:.3f}<extra>Surface</extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[current_m],
            y=[current_b],
            z=[current_loss],
            mode="markers",
            name="Current parameters",
            marker=dict(size=7, color="#ef4444", symbol="diamond"),
            hovertemplate="m=%{x:.2f}<br>b=%{y:.2f}<br>loss=%{z:.3f}<extra>Current</extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[min_m],
            y=[min_b],
            z=[min_loss],
            mode="markers",
            name="Grid minimum",
            marker=dict(size=8, color="#f59e0b", symbol="circle"),
            hovertemplate="m=%{x:.2f}<br>b=%{y:.2f}<br>loss=%{z:.3f}<extra>Minimum</extra>",
        )
    )
    fig.update_layout(
        title="3D Loss Surface",
        height=560,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title="Slope (m)",
            yaxis_title="Intercept (b)",
            zaxis_title="Loss",
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.9)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_contour(
    m_grid: np.ndarray,
    b_grid: np.ndarray,
    losses: np.ndarray,
    current_point: tuple[float, float, float],
    global_minimum: tuple[float, float, float],
    gd_path: Sequence[tuple[float, float, float]] | None = None,
) -> go.Figure:
    """Render a contour plot with current point, minimum, and optional GD path."""
    current_m, current_b, _ = current_point
    min_m, min_b, _ = global_minimum

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=m_grid[0],
            y=b_grid[:, 0],
            z=losses,
            colorscale="Viridis",
            contours=dict(showlabels=True),
            colorbar=dict(title="MSE"),
            hovertemplate="m=%{x:.2f}<br>b=%{y:.2f}<br>loss=%{z:.3f}<extra>Contour</extra>",
        )
    )

    if gd_path:
        m_path = [point[0] for point in gd_path]
        b_path = [point[1] for point in gd_path]
        fig.add_trace(
            go.Scatter(
                x=m_path,
                y=b_path,
                mode="lines+markers",
                name="Gradient descent path",
                line=dict(color="#111827", width=3),
                marker=dict(size=6, color="#111827"),
                hovertemplate="m=%{x:.2f}<br>b=%{y:.2f}<extra>GD step</extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[current_m],
            y=[current_b],
            mode="markers",
            name="Current parameters",
            marker=dict(size=13, color="#ef4444", symbol="x"),
            hovertemplate="m=%{x:.2f}<br>b=%{y:.2f}<extra>Current</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_m],
            y=[min_b],
            mode="markers",
            name="Grid minimum",
            marker=dict(size=13, color="#f59e0b", symbol="star"),
            hovertemplate="m=%{x:.2f}<br>b=%{y:.2f}<extra>Minimum</extra>",
        )
    )
    fig.update_layout(
        title="Contour Loss Landscape",
        template="plotly_white",
        height=560,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Slope (m)",
        yaxis_title="Intercept (b)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_heatmap(
    m_grid: np.ndarray,
    b_grid: np.ndarray,
    losses: np.ndarray,
    current_point: tuple[float, float, float],
    global_minimum: tuple[float, float, float],
) -> go.Figure:
    """Render a heatmap representation of the loss matrix."""
    current_m, current_b, _ = current_point
    min_m, min_b, _ = global_minimum

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=m_grid[0],
            y=b_grid[:, 0],
            z=losses,
            colorscale="Viridis",
            colorbar=dict(title="MSE"),
            hovertemplate="m=%{x:.2f}<br>b=%{y:.2f}<br>loss=%{z:.3f}<extra>Heatmap</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_m],
            y=[current_b],
            mode="markers",
            name="Current",
            marker=dict(size=11, color="#ef4444", symbol="x"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_m],
            y=[min_b],
            mode="markers",
            name="Minimum",
            marker=dict(size=11, color="#f59e0b", symbol="star"),
        )
    )
    fig.update_layout(
        title="Loss Heatmap",
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Slope (m)",
        yaxis_title="Intercept (b)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def render_learning_panel() -> None:
    """Show short educational explanations in the sidebar."""
    st.sidebar.markdown("## Learning Panel")
    st.sidebar.info(
        "A loss surface maps every parameter choice `(m, b)` to a model error. "
        "Low regions correspond to better-fitting lines."
    )
    st.sidebar.success(
        "For linear regression with MSE, the surface is convex. That matters because "
        "there is one broad basin leading to a global minimum rather than many local traps."
    )
    st.sidebar.warning(
        "Gradient descent follows the slope of the surface downhill. The learning rate "
        "controls step size: too small is slow, too large can overshoot."
    )


def render_header() -> None:
    """Render the title and short intro copy."""
    st.title(APP_TITLE)
    st.markdown(
        """
        Explore how a simple linear model changes as you move through parameter space.
        This tool connects three ideas in one view: the dataset, the loss landscape, and
        the optimization path taken by gradient descent.
        """
    )


def main() -> None:
    """Build the full Streamlit application."""
    initialize_state()
    render_learning_panel()

    st.sidebar.markdown("## Controls")
    st.sidebar.button("Reset parameters", on_click=reset_controls, use_container_width=True)

    st.session_state["m"] = st.sidebar.slider(
        "Slope (m)",
        min_value=M_RANGE[0],
        max_value=M_RANGE[1],
        value=float(st.session_state["m"]),
        step=0.1,
    )
    st.session_state["b"] = st.sidebar.slider(
        "Intercept (b)",
        min_value=B_RANGE[0],
        max_value=B_RANGE[1],
        value=float(st.session_state["b"]),
        step=0.1,
    )
    st.session_state["learning_rate"] = st.sidebar.slider(
        "Learning rate",
        min_value=0.01,
        max_value=0.20,
        value=float(st.session_state["learning_rate"]),
        step=0.01,
    )
    st.session_state["gd_steps"] = st.sidebar.slider(
        "Gradient descent steps",
        min_value=5,
        max_value=50,
        value=int(st.session_state["gd_steps"]),
        step=1,
    )
    st.session_state["show_gd"] = st.sidebar.toggle(
        "Show gradient descent",
        value=bool(st.session_state["show_gd"]),
    )
    st.session_state["animate_gd"] = st.sidebar.toggle(
        "Animate descent",
        value=bool(st.session_state["animate_gd"]),
    )
    st.session_state["gd_seed"] = st.sidebar.slider(
        "GD start seed",
        min_value=0,
        max_value=50,
        value=int(st.session_state["gd_seed"]),
        step=1,
    )
    st.session_state["data_seed"] = st.sidebar.slider(
        "Dataset seed",
        min_value=0,
        max_value=50,
        value=int(st.session_state["data_seed"]),
        step=1,
    )
    st.session_state["noise_scale"] = st.sidebar.slider(
        "Noise level",
        min_value=0.2,
        max_value=2.5,
        value=float(st.session_state["noise_scale"]),
        step=0.1,
    )

    render_header()

    x, y = generate_data(
        num_points=DEFAULT_POINTS,
        noise_scale=float(st.session_state["noise_scale"]),
        seed=int(st.session_state["data_seed"]),
    )
    m_grid, b_grid, losses = compute_loss_surface(x, y, resolution=GRID_SIZE)

    current_m = float(st.session_state["m"])
    current_b = float(st.session_state["b"])
    current_loss = compute_loss(x, y, current_m, current_b)
    current_point = (current_m, current_b, current_loss)
    global_minimum = get_global_minimum(m_grid, b_grid, losses)

    rng = np.random.default_rng(int(st.session_state["gd_seed"]))
    start_m = float(rng.uniform(M_RANGE[0] + 0.5, M_RANGE[1] - 0.5))
    start_b = float(rng.uniform(B_RANGE[0] + 0.5, B_RANGE[1] - 0.5))
    gd_path = None
    if st.session_state["show_gd"]:
        gd_path = gradient_descent(
            x,
            y,
            start_m=start_m,
            start_b=start_b,
            learning_rate=float(st.session_state["learning_rate"]),
            steps=int(st.session_state["gd_steps"]),
        )

    st.markdown("## Dataset Visualization")
    st.caption(
        "The scatter plot shows synthetic observations. The red line is the model generated "
        "from your current parameter choices."
    )
    dataset_col, metrics_col = st.columns([2.3, 1.2], gap="large")

    with dataset_col:
        st.plotly_chart(plot_dataset(x, y, current_m, current_b), use_container_width=True)

    with metrics_col:
        st.markdown("### Model Snapshot")
        st.metric("Current loss", f"{current_loss:.4f}")
        st.metric("Current slope (m)", f"{current_m:.2f}")
        st.metric("Current intercept (b)", f"{current_b:.2f}")
        st.metric("Grid minimum loss", f"{global_minimum[2]:.4f}")
        st.metric("Best grid m", f"{global_minimum[0]:.2f}")
        st.metric("Best grid b", f"{global_minimum[1]:.2f}")
        st.download_button(
            "Download dataset CSV",
            data=dataset_to_csv(x, y),
            file_name="loss_surface_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("## Loss Surface Visualization")
    st.caption(
        "Each point on these plots represents one candidate line. The highlighted markers show "
        "your current choice and the lowest point found on the evaluation grid."
    )

    surface_col, contour_col = st.columns(2, gap="large")
    with surface_col:
        st.plotly_chart(
            plot_surface(m_grid, b_grid, losses, current_point, global_minimum),
            use_container_width=True,
        )
    with contour_col:
        contour_placeholder = st.empty()
        if gd_path and st.session_state["animate_gd"]:
            for step_index in range(1, len(gd_path) + 1):
                contour_placeholder.plotly_chart(
                    plot_contour(
                        m_grid,
                        b_grid,
                        losses,
                        current_point,
                        global_minimum,
                        gd_path[:step_index],
                    ),
                    use_container_width=True,
                )
        else:
            contour_placeholder.plotly_chart(
                plot_contour(m_grid, b_grid, losses, current_point, global_minimum, gd_path),
                use_container_width=True,
            )

    st.markdown("## Optimization Demo")
    explainer_col, heatmap_col = st.columns([1.15, 1.85], gap="large")

    with explainer_col:
        st.markdown(
            """
            **What you are seeing**

            `J(m, b)` is the Mean Squared Error for the current line.

            A convex loss surface means there is one dominant valley, so optimization is easier
            to understand and more predictable.

            When gradient descent is enabled, the dark path shows how repeated parameter updates
            move toward lower loss.
            """
        )
        if gd_path:
            start_loss = gd_path[0][2]
            end_loss = gd_path[-1][2]
            st.metric("GD start loss", f"{start_loss:.4f}")
            st.metric("GD final loss", f"{end_loss:.4f}")
            st.metric("Loss improvement", f"{start_loss - end_loss:.4f}")
            st.caption(
                f"Gradient descent starts at m={start_m:.2f}, b={start_b:.2f} "
                f"with learning rate {st.session_state['learning_rate']:.2f}."
            )
        else:
            st.info("Enable gradient descent in the sidebar to visualize the optimization path.")

    with heatmap_col:
        st.plotly_chart(
            plot_heatmap(m_grid, b_grid, losses, current_point, global_minimum),
            use_container_width=True,
        )

    st.markdown("## Why This Matters")
    insight_cols = st.columns(3, gap="large")
    insight_cols[0].markdown(
        """
        **Parameter intuition**

        Moving `m` changes tilt.
        Moving `b` shifts the line vertically.
        """
    )
    insight_cols[1].markdown(
        """
        **Optimization intuition**

        Good optimization follows the surface downhill without taking unstable jumps.
        """
    )
    insight_cols[2].markdown(
        """
        **Modeling intuition**

        The lowest-loss region corresponds to the line that best explains the observed data.
        """
    )


if __name__ == "__main__":
    main()
