# Loss Surface Visualizer

An interactive Streamlit app that helps users build intuition for optimization in linear regression by visualizing how slope (`m`) and intercept (`b`) change model loss.

## Description

Linear regression is often introduced with a single best-fit line, but the real learning happens when you can see how different parameter choices affect error. This project visualizes the loss surface for a simple linear model and shows how gradient descent moves through parameter space toward a minimum.

## Features

- Interactive controls for slope, intercept, learning rate, and gradient descent steps
- Synthetic noisy linear dataset with live regression line updates
- 3D Plotly surface plot for the loss landscape
- Contour plot for top-down optimization intuition
- Heatmap view for quick comparison of loss values
- Current parameter marker and grid-based global minimum marker
- Gradient descent path visualization with optional animation
- Download button for the generated dataset
- Sidebar learning panel with concise educational explanations

## Demo Instructions

1. Use the sidebar sliders to change `m` and `b`.
2. Watch how the regression line shifts on the dataset plot.
3. Compare the current loss against the global minimum shown on the grid.
4. Turn on gradient descent to see how optimization moves across the contour map.
5. Enable animation for a step-by-step view of descent.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Instructions

1. Push this project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in and connect your GitHub account.
4. Select this repository.
5. Choose `app.py` as the main file.
6. Deploy the app.

## Project Structure

```text
.
├── app.py
├── requirements.txt
└── README.md
```

## Screenshots

- `docs/screenshot-home.png` - Add an overview screenshot of the main dashboard
- `docs/screenshot-contour.png` - Add a screenshot of the contour plot with gradient descent

## Tech Stack

- Streamlit
- NumPy
- Plotly

## License

This project is available for educational and portfolio use.
