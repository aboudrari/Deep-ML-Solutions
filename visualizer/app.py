import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_regression, make_blobs, make_moons, make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

st.set_page_config(page_title="ML Algorithm Explorer", layout="wide", page_icon="🧠")

# ── Roadmaps ──────────────────────────────────────────────────────────────────

ROADMAPS = {
    "KNN": [
        {
            "step": 1,
            "title": "Choose K (number of neighbors)",
            "description": "K controls how many neighbors vote on the prediction. Small K → complex, wiggly boundary. Large K → smooth, simple boundary.",
            "math": r"K \in \{1, 2, 3, \ldots, n\}",
            "code": "knn = KNeighborsClassifier(n_neighbors=K)",
        },
        {
            "step": 2,
            "title": "Compute distances to all training points",
            "description": "For a new query point, calculate its distance to every point in the training set. Euclidean distance is the default.",
            "math": r"d(x,\, x_i) = \sqrt{\sum_{j=1}^{p}(x_j - x_{ij})^2}",
            "code": "distances = np.sqrt(np.sum((X_train - x_query)**2, axis=1))",
        },
        {
            "step": 3,
            "title": "Find the K nearest neighbors",
            "description": "Sort distances and pick the K smallest — those are the neighbors that will vote.",
            "math": r"N_K(x) = \text{K points with smallest } d(x, x_i)",
            "code": "k_indices = np.argsort(distances)[:K]",
        },
        {
            "step": 4,
            "title": "Vote (classification) or Average (regression)",
            "description": "Classification → majority class wins. Regression → average of neighbor values.",
            "math": r"\hat{y} = \text{mode}\!\left(y_{N_K(x)}\right)",
            "code": "prediction = np.bincount(y_train[k_indices]).argmax()",
        },
    ],
    "Decision Tree": [
        {
            "step": 1,
            "title": "Start at the root with all data",
            "description": "The full dataset enters the root node. The goal is to find which feature + threshold splits the data best.",
            "math": r"D = \{(x_1, y_1),\ldots,(x_n, y_n)\}",
            "code": "tree = DecisionTreeClassifier(max_depth=max_depth)",
        },
        {
            "step": 2,
            "title": "Find the best split (Gini / Entropy)",
            "description": "Try every feature and every threshold. Pick the combination that reduces impurity the most.",
            "math": r"Gini = 1 - \sum_{k} p_k^2",
            "code": "best_split = min(splits, key=lambda s: gini(s))",
        },
        {
            "step": 3,
            "title": "Split the data into two child nodes",
            "description": "Divide samples: left child gets values ≤ threshold, right child gets values > threshold.",
            "math": r"D_L = \{x \mid x_j \leq t\},\quad D_R = \{x \mid x_j > t\}",
            "code": "left  = X[X[:, feature] <= threshold]\nright = X[X[:, feature] >  threshold]",
        },
        {
            "step": 4,
            "title": "Recurse until a stopping condition",
            "description": "Repeat steps 2–3 for each child until max_depth is reached or the node is pure (all same class).",
            "math": r"\text{Stop if: depth} = d_{\max}\ \text{or}\ |D| < \text{min\_samples}",
            "code": "if depth < max_depth and not pure:\n    split(node)",
        },
        {
            "step": 5,
            "title": "Predict by traversing the tree",
            "description": "For a new point, follow left/right branches from root to a leaf. The leaf's majority class is the prediction.",
            "math": r"\hat{y} = \text{class at leaf}(x)",
            "code": "prediction = tree.predict([x_new])",
        },
    ],
    "Linear Regression": [
        {
            "step": 1,
            "title": "Define the model",
            "description": "We want a line (or hyperplane) that best fits the data: output = weight × input + bias.",
            "math": r"\hat{y} = w \cdot x + b",
            "code": "y_pred = X @ weights + bias",
        },
        {
            "step": 2,
            "title": "Define the cost function (MSE)",
            "description": "Measure how wrong the predictions are. MSE penalises large errors heavily.",
            "math": r"\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2",
            "code": "loss = np.mean((y - y_pred)**2)",
        },
        {
            "step": 3,
            "title": "Compute gradients",
            "description": "Find the direction (gradient) that tells us how to tweak weights to reduce the loss.",
            "math": r"\frac{\partial \mathcal{L}}{\partial w} = \frac{-2}{n}\sum x_i(y_i - \hat{y}_i)",
            "code": "dw = -2/n * X.T @ (y - y_pred)\ndb = -2/n * np.sum(y - y_pred)",
        },
        {
            "step": 4,
            "title": "Update weights (Gradient Descent)",
            "description": "Move weights in the opposite direction of the gradient, scaled by the learning rate α.",
            "math": r"w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}",
            "code": "weights -= lr * dw\nbias   -= lr * db",
        },
        {
            "step": 5,
            "title": "Repeat until convergence",
            "description": "Keep iterating until the loss barely changes between steps.",
            "math": r"\text{Repeat until }|L_t - L_{t-1}| < \varepsilon",
            "code": "for epoch in range(n_iterations):\n    # steps 2 → 4",
        },
    ],
    "Logistic Regression": [
        {
            "step": 1,
            "title": "Compute the linear combination",
            "description": "Same start as linear regression: a weighted sum of the input features.",
            "math": r"z = w \cdot x + b",
            "code": "z = X @ weights + bias",
        },
        {
            "step": 2,
            "title": "Apply Sigmoid to get a probability",
            "description": "The sigmoid squashes z into (0, 1) so it can be interpreted as a probability.",
            "math": r"\sigma(z) = \frac{1}{1 + e^{-z}}",
            "code": "y_pred = 1 / (1 + np.exp(-z))",
        },
        {
            "step": 3,
            "title": "Compute Binary Cross-Entropy Loss",
            "description": "Penalise confident wrong predictions more than uncertain ones.",
            "math": r"\mathcal{L} = -\frac{1}{n}\sum\!\left[y\log\hat{y} + (1-y)\log(1-\hat{y})\right]",
            "code": "loss = -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))",
        },
        {
            "step": 4,
            "title": "Gradient Descent to minimise loss",
            "description": "The gradient for logistic regression is clean: just the error times the input.",
            "math": r"\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^T(\hat{y} - y)",
            "code": "dw = (1/n) * X.T @ (y_pred - y)\nweights -= lr * dw",
        },
        {
            "step": 5,
            "title": "Predict: threshold at 0.5",
            "description": "If the predicted probability is ≥ 0.5 → class 1, otherwise → class 0.",
            "math": r"\hat{y} = \begin{cases}1 & \sigma(z) \geq 0.5 \\ 0 & \text{otherwise}\end{cases}",
            "code": "predictions = (y_pred >= 0.5).astype(int)",
        },
    ],
}

# ── Dataset helpers ────────────────────────────────────────────────────────────

def get_classification_data(shape, n_samples, noise, seed):
    if shape == "Blobs":
        return make_blobs(n_samples=n_samples, centers=2, random_state=seed)
    if shape == "Moons":
        return make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    if shape == "Circles":
        return make_circles(n_samples=n_samples, noise=noise, random_state=seed)
    return make_classification(n_samples=n_samples, n_features=2, n_redundant=0, random_state=seed)


def get_regression_data(n_samples, noise, seed):
    X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise * 80, random_state=seed)
    return X, y

# ── Plot helpers ───────────────────────────────────────────────────────────────

CMAP_BG   = ListedColormap(["#FFCCCC", "#CCCCFF"])
COLORS    = ["#E63946", "#1D3557"]


def plot_decision_boundary(model, X, y, ax, title=""):
    h = 0.03
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=CMAP_BG, alpha=0.55)
    for cls, color in enumerate(COLORS):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=color, label=f"Class {cls}",
                   edgecolors="white", linewidths=0.5, s=45, zorder=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)


def plot_sigmoid(ax):
    z = np.linspace(-7, 7, 300)
    sig = 1 / (1 + np.exp(-z))
    ax.plot(z, sig, color="#6A0DAD", linewidth=2.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold = 0.5")
    ax.fill_between(z, sig, 0.5, where=(sig >= 0.5), alpha=0.18, color=COLORS[1], label="→ Class 1")
    ax.fill_between(z, sig, 0.5, where=(sig <  0.5), alpha=0.18, color=COLORS[0], label="→ Class 0")
    ax.set_xlabel("z = wx + b", fontsize=10)
    ax.set_ylabel("σ(z)", fontsize=10)
    ax.set_title("Sigmoid Function", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.title("🧠 ML Algorithm Explorer")
    st.markdown("*Pick an algorithm → follow the roadmap → explore the live visualisation.*")

    # ── Sidebar ──
    st.sidebar.title("⚙️ Controls")
    algo = st.sidebar.selectbox(
        "Algorithm",
        ["KNN", "Decision Tree (Classification)", "Decision Tree (Regression)",
         "Linear Regression", "Logistic Regression"],
    )

    roadmap_key = (
        "Decision Tree"  if "Decision Tree"  in algo else
        "Logistic Regression" if algo == "Logistic Regression" else
        algo
    )

    is_classification = algo not in ["Decision Tree (Regression)", "Linear Regression"]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset")
    n_samples = st.sidebar.slider("Samples", 50, 600, 200, step=50)
    noise     = st.sidebar.slider("Noise",   0.0,  0.5, 0.15, step=0.05)
    if is_classification:
        shape = st.sidebar.selectbox("Shape", ["Blobs", "Moons", "Circles", "Linear"])
    seed = st.sidebar.slider("Random Seed", 0, 100, 42)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hyperparameters")

    if algo == "KNN":
        k      = st.sidebar.slider("K (neighbors)", 1, 25, 5)
        metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "minkowski"])

    elif "Decision Tree" in algo:
        max_depth    = st.sidebar.slider("Max Depth", 1, 12, 3)
        min_samples  = st.sidebar.slider("Min Samples Split", 2, 30, 2)
        if "Classification" in algo:
            criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        else:
            criterion = st.sidebar.selectbox("Criterion", ["squared_error", "absolute_error"])

    elif algo == "Linear Regression":
        show_residuals = st.sidebar.checkbox("Show Residuals", value=True)

    elif algo == "Logistic Regression":
        C = st.sidebar.slider("C (Regularisation)", 0.01, 10.0, 1.0, step=0.1)

    # ── Tabs ──
    tab_road, tab_viz = st.tabs(["📋 Roadmap", "📊 Visualisation"])

    # ── Roadmap Tab ──
    with tab_road:
        st.header(f"{algo} — Step-by-Step Roadmap")
        for step in ROADMAPS[roadmap_key]:
            with st.expander(f"Step {step['step']}: {step['title']}", expanded=True):
                st.write(step["description"])
                st.latex(step["math"])
                st.code(step["code"], language="python")

    # ── Visualisation Tab ──
    with tab_viz:
        st.header(f"{algo} — Live Visualisation")

        # Generate data
        if is_classification:
            X, y = get_classification_data(shape, n_samples, noise, seed)
        else:
            X, y = get_regression_data(n_samples, noise, seed)

        # ── KNN ──
        if algo == "KNN":
            model = KNeighborsClassifier(n_neighbors=k, metric=metric)
            model.fit(X, y)
            acc = accuracy_score(y, model.predict(X))

            st.metric("Training Accuracy", f"{acc:.2%}")
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_decision_boundary(model, X, y, ax, f"KNN  |  K={k}  |  metric={metric}")
            st.pyplot(fig)
            plt.close()

        # ── Decision Tree Classification ──
        elif algo == "Decision Tree (Classification)":
            model = DecisionTreeClassifier(
                max_depth=max_depth, min_samples_split=min_samples,
                criterion=criterion, random_state=42)
            model.fit(X, y)
            acc = accuracy_score(y, model.predict(X))

            st.metric("Training Accuracy", f"{acc:.2%}")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(7, 5))
                plot_decision_boundary(model, X, y, ax, f"Decision Boundary  |  depth={max_depth}")
                st.pyplot(fig); plt.close()
            with col2:
                fig, ax = plt.subplots(figsize=(7, 5))
                plot_tree(model, ax=ax, filled=True, feature_names=["x₁", "x₂"],
                          class_names=["0", "1"], rounded=True, fontsize=8)
                ax.set_title("Tree Structure", fontsize=12, fontweight="bold")
                st.pyplot(fig); plt.close()

        # ── Decision Tree Regression ──
        elif algo == "Decision Tree (Regression)":
            model = DecisionTreeRegressor(
                max_depth=max_depth, min_samples_split=min_samples,
                criterion=criterion, random_state=42)
            model.fit(X, y)

            X_line = np.linspace(X.min(), X.max(), 400).reshape(-1, 1)
            y_line = model.predict(X_line)

            mse = mean_squared_error(y, model.predict(X))
            r2  = r2_score(y, model.predict(X))
            col1, col2 = st.columns(2)
            col1.metric("MSE",      f"{mse:.2f}")
            col2.metric("R² Score", f"{r2:.4f}")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X, y, alpha=0.5, color="steelblue", s=30, label="Data")
            ax.plot(X_line, y_line, color="#E63946", linewidth=2.5, label="Tree prediction")
            ax.set_title(f"Decision Tree Regression  |  depth={max_depth}", fontsize=12, fontweight="bold")
            ax.legend()
            st.pyplot(fig); plt.close()

        # ── Linear Regression ──
        elif algo == "Linear Regression":
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            mse = mean_squared_error(y, y_pred)
            r2  = r2_score(y, y_pred)
            col1, col2 = st.columns(2)
            col1.metric("MSE",      f"{mse:.2f}")
            col2.metric("R² Score", f"{r2:.4f}")

            X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
            y_line = model.predict(X_line)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X, y, alpha=0.5, color="steelblue", s=30, label="Data")
            ax.plot(X_line, y_line, color="#E63946", linewidth=2.5,
                    label=f"ŷ = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
            if show_residuals:
                for xi, yi, yp in zip(X.ravel(), y, y_pred):
                    ax.plot([xi, xi], [yi, yp], color="gray", alpha=0.35, linewidth=0.8)
            ax.set_title("Linear Regression", fontsize=12, fontweight="bold")
            ax.legend()
            st.pyplot(fig); plt.close()

        # ── Logistic Regression ──
        elif algo == "Logistic Regression":
            model = LogisticRegression(C=C, random_state=42, max_iter=1000)
            model.fit(X, y)
            acc = accuracy_score(y, model.predict(X))

            st.metric("Training Accuracy", f"{acc:.2%}")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(7, 5))
                plot_decision_boundary(model, X, y, ax, f"Decision Boundary  |  C={C}")
                st.pyplot(fig); plt.close()
            with col2:
                fig, ax = plt.subplots(figsize=(7, 5))
                plot_sigmoid(ax)
                st.pyplot(fig); plt.close()


if __name__ == "__main__":
    main()
