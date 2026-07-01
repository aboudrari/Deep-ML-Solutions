import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gradient Descent (Linear Regression from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def _gd_gen(seed):
    rng  = np.random.default_rng(seed)
    n    = 30
    X    = rng.uniform(-3, 3, n)
    y    = 2.5 * X + 1.0 + rng.normal(0, 0.8, n)
    lr   = 0.05
    iters = 80
    w, b = 0.0, 0.0
    losses = []
    ws, bs = [w], [b]
    for _ in range(iters):
        y_pred = w * X + b
        loss   = np.mean((y - y_pred)**2)
        losses.append(loss)
        dw = -2 * np.mean(X * (y - y_pred))
        db = -2 * np.mean(y - y_pred)
        w -= lr * dw
        b -= lr * db
        ws.append(w); bs.append(b)
    return {
        "inputs": {"learning rate": lr, "iterations": iters, "n samples": n},
        "X": X, "y": y, "losses": losses,
        "w_final": round(w, 4), "b_final": round(b, 4),
        "ws": ws, "bs": bs,
        "answer": f"w={round(w,4)}, b={round(b,4)}",
    }

def _gd_steps(ex):
    return [
        {
            "title": "Initialise weights to zero",
            "explanation": "Start with w=0 and b=0 — gradient descent will find the right values.",
            "math": r"w \leftarrow 0,\quad b \leftarrow 0",
            "code": "w, b = 0.0, 0.0",
        },
        {
            "title": "Compute predictions and MSE loss",
            "explanation": "Forward pass: ŷ = wx + b. Loss = mean squared error.",
            "math": r"\mathcal{L} = \frac{1}{n}\sum(y - wx - b)^2",
            "code": "y_pred = w * X + b\nloss = np.mean((y - y_pred)**2)",
        },
        {
            "title": "Compute gradients",
            "explanation": "Partial derivatives of the loss with respect to w and b.",
            "math": r"\frac{\partial L}{\partial w} = -\frac{2}{n}\sum x(y-\hat{y}), \quad \frac{\partial L}{\partial b} = -\frac{2}{n}\sum(y-\hat{y})",
            "code": "dw = -2 * np.mean(X * (y - y_pred))\ndb = -2 * np.mean(y - y_pred)",
        },
        {
            "title": "Update weights",
            "explanation": f"Move weights in the opposite direction of the gradient, scaled by lr={ex['inputs']['learning rate']}.",
            "math": r"w \leftarrow w - \alpha \frac{\partial L}{\partial w}, \quad b \leftarrow b - \alpha \frac{\partial L}{\partial b}",
            "code": "w -= lr * dw\nb -= lr * db",
        },
        {
            "title": "Converged result",
            "explanation": f"After {ex['inputs']['iterations']} iterations: w = {ex['w_final']}, b = {ex['b_final']}. Loss dropped from {ex['losses'][0]:.2f} → {ex['losses'][-1]:.2f}",
            "result": ex["answer"],
        },
    ]

def _gd_viz(ex):
    X, y = ex["X"], ex["y"]
    ws, bs, losses = ex["ws"], ex["bs"], ex["losses"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.scatter(X, y, color="#1D3557", s=40, alpha=0.7, label="Data")
    x_line = np.linspace(X.min(), X.max(), 100)
    ax1.plot(x_line, ws[0] * x_line + bs[0],  "#CCCCCC", lw=1.5, label="Initial (w=0, b=0)")
    ax1.plot(x_line, ws[-1] * x_line + bs[-1], "#E63946", lw=2.5, label=f"Final (w={ex['w_final']}, b={ex['b_final']})")
    for i in [5, 15, 30]:
        ax1.plot(x_line, ws[i] * x_line + bs[i], "#2A9D8F", lw=1, alpha=0.4)
    ax1.set_title("Regression Line: Initial → Final", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(losses, "#E63946", lw=2.5)
    ax2.set_title(f"Loss Curve ({len(losses)} iterations)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("MSE Loss")
    ax2.grid(True, alpha=0.3)
    ax2.text(len(losses) * 0.6, losses[0] * 0.8, f"Start: {losses[0]:.2f}", fontsize=9, color="gray")
    ax2.text(len(losses) * 0.7, losses[-1] + losses[0]*0.05, f"End: {losses[-1]:.2f}", fontsize=9, color="#E63946")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. L2 Regularisation (Weight Decay)
# ─────────────────────────────────────────────────────────────────────────────

def _l2_gen(seed):
    rng     = np.random.default_rng(seed)
    n, d    = 20, 5
    X       = rng.normal(0, 1, (n, d))
    true_w  = rng.normal(0, 2, d)
    y       = X @ true_w + rng.normal(0, 0.5, n)
    lambdas = [0, 0.1, 1.0, 10.0]
    weights = {}
    for lam in lambdas:
        # closed form: w = (X'X + λI)^-1 X'y
        w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
        weights[lam] = np.round(w, 4)
    return {
        "inputs": {"n samples": n, "n features": d, "λ values tested": lambdas},
        "X": X, "y": y,
        "lambdas": lambdas,
        "weights": weights,
        "answer": f"As λ increases, weights shrink toward 0",
    }

def _l2_steps(ex):
    return [
        {
            "title": "Standard loss (no regularisation)",
            "explanation": "Minimise MSE alone — can lead to large weights and overfitting.",
            "math": r"\mathcal{L} = \frac{1}{n}\|y - Xw\|^2",
            "code": "w = np.linalg.lstsq(X, y, rcond=None)[0]",
        },
        {
            "title": "Add the L2 penalty term",
            "explanation": "Add λ||w||² to the loss. This penalises large weights.",
            "math": r"\mathcal{L}_{\text{L2}} = \frac{1}{n}\|y - Xw\|^2 + \lambda\|w\|^2",
            "code": "loss_l2 = mse_loss + lambda_ * np.sum(w**2)",
        },
        {
            "title": "Closed-form solution with L2",
            "explanation": "The L2 term modifies the normal equations by adding λI to X'X.",
            "math": r"w^* = (X^TX + \lambda I)^{-1} X^T y",
            "code": "w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)",
        },
        {
            "title": "Effect: larger λ → smaller weights",
            "explanation": "As λ increases, weights are pushed toward 0. At λ=0, it's ordinary least squares.",
            "result": ex["answer"],
        },
    ]

def _l2_viz(ex):
    lambdas = ex["lambdas"]
    weights = ex["weights"]
    n_feats = len(next(iter(weights.values())))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cmap = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))
    for i, lam in enumerate(lambdas):
        w = weights[lam]
        ax1.plot(range(n_feats), w, "o-", color=cmap[i], lw=2, label=f"λ={lam}")
    ax1.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax1.set_title("Weight Values at Different λ", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Feature index"); ax1.set_ylabel("Weight value")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    norms = [float(np.linalg.norm(weights[lam])) for lam in lambdas]
    ax2.plot([str(l) for l in lambdas], norms, "o-", color="#E63946", lw=2.5, markersize=10)
    ax2.set_title("||w|| vs λ (regularisation strength)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("λ (lambda)"); ax2.set_ylabel("||w|| (L2 norm)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. K-Fold Cross Validation
# ─────────────────────────────────────────────────────────────────────────────

def _kfold_gen(seed):
    rng  = np.random.default_rng(seed)
    n, k = 60, 5
    X    = rng.normal(0, 1, (n, 2))
    y    = 3 * X[:, 0] + rng.normal(0, 1, n)
    kf   = KFold(n_splits=k, shuffle=True, random_state=int(seed))
    scores = []
    splits = []
    for train_idx, val_idx in kf.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        val_pred = model.predict(X[val_idx])
        mse = mean_squared_error(y[val_idx], val_pred)
        scores.append(round(mse, 4))
        splits.append((train_idx.tolist(), val_idx.tolist()))
    mean_score = round(float(np.mean(scores)), 4)
    std_score  = round(float(np.std(scores)),  4)
    return {
        "inputs": {"n samples": n, "k folds": k},
        "k": k, "scores": scores, "splits": splits,
        "mean": mean_score, "std": std_score,
        "answer": f"Mean MSE = {mean_score} ± {std_score}",
    }

def _kfold_steps(ex):
    k, scores = ex["k"], ex["scores"]
    return [
        {
            "title": "Split data into K equal folds",
            "explanation": f"Divide the dataset into {k} non-overlapping folds of roughly equal size.",
            "math": r"D = F_1 \cup F_2 \cup \cdots \cup F_k, \quad F_i \cap F_j = \emptyset",
            "code": "kf = KFold(n_splits=k, shuffle=True, random_state=42)",
        },
        {
            "title": "Train and validate K times",
            "explanation": f"Each fold takes a turn as the validation set. The other {k-1} folds are training data.",
            "code": "for train_idx, val_idx in kf.split(X):\n    model.fit(X[train_idx], y[train_idx])\n    score = mse(y[val_idx], model.predict(X[val_idx]))",
        },
        {
            "title": "Collect fold scores",
            "explanation": f"MSE per fold: {scores}",
            "result": scores,
        },
        {
            "title": "Average the scores",
            "explanation": "The mean and std give a reliable estimate of model performance.",
            "math": r"\bar{\text{score}} = \frac{1}{k}\sum_{i=1}^{k}\text{score}_i",
            "result": f"Mean={ex['mean']},  Std={ex['std']}",
            "code": "mean_score = np.mean(scores)\nstd_score  = np.std(scores)",
        },
    ]

def _kfold_viz(ex):
    k, scores = ex["k"], ex["scores"]
    splits    = ex["splits"]
    n         = len(splits[0][0]) + len(splits[0][1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    colors = plt.cm.Set2(np.linspace(0, 1, k))
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        ax1.scatter(train_idx, [fold_i] * len(train_idx), color=colors[fold_i], s=20, alpha=0.4, marker="|")
        ax1.scatter(val_idx,   [fold_i] * len(val_idx),   color="#E63946",       s=20, marker="|")
    ax1.set_yticks(range(k)); ax1.set_yticklabels([f"Fold {i+1}" for i in range(k)])
    ax1.set_xlabel("Sample index")
    ax1.set_title("K-Fold Split Structure\n(red = validation, colored = train)", fontsize=11, fontweight="bold")

    bars = ax2.bar(range(1, k + 1), scores, color=colors, edgecolor="white", alpha=0.9)
    ax2.axhline(ex["mean"], color="#E63946", lw=2, linestyle="--", label=f"Mean = {ex['mean']}")
    ax2.fill_between(
        [0.5, k + 0.5],
        ex["mean"] - ex["std"], ex["mean"] + ex["std"],
        alpha=0.15, color="#E63946", label=f"± std = {ex['std']}"
    )
    ax2.set_xticks(range(1, k + 1)); ax2.set_xticklabels([f"Fold {i}" for i in range(1, k + 1)])
    ax2.set_ylabel("MSE"); ax2.set_title("Validation MSE per Fold", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    for bar, s in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, s + 0.01, f"{s:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

def _early_stop_gen(seed):
    rng    = np.random.default_rng(seed)
    epochs = 80
    t      = np.arange(epochs)
    train_loss = 2.0 * np.exp(-0.07 * t) + 0.1 + rng.normal(0, 0.02, epochs)
    val_loss   = 2.0 * np.exp(-0.05 * t) + 0.3 + 0.004 * t + rng.normal(0, 0.04, epochs)
    patience   = 8
    best_val   = float("inf")
    best_epoch = 0
    wait       = 0
    stop_epoch = epochs - 1
    for ep in range(epochs):
        if val_loss[ep] < best_val:
            best_val   = val_loss[ep]
            best_epoch = ep
            wait       = 0
        else:
            wait += 1
            if wait >= patience:
                stop_epoch = ep
                break
    return {
        "inputs": {"patience": patience, "epochs run": epochs},
        "train_loss": train_loss, "val_loss": val_loss,
        "best_epoch": best_epoch, "stop_epoch": stop_epoch, "patience": patience,
        "answer": f"Stop at epoch {stop_epoch} (best val at epoch {best_epoch})",
    }

def _early_stop_steps(ex):
    return [
        {
            "title": "Monitor validation loss each epoch",
            "explanation": "After each epoch, check if the validation loss improved.",
            "code": "if val_loss < best_val:\n    best_val = val_loss; wait = 0\nelse:\n    wait += 1",
        },
        {
            "title": "Reset counter on improvement",
            "explanation": f"Best validation loss found at epoch {ex['best_epoch']}. Reset patience counter to 0.",
            "result": f"Best val epoch = {ex['best_epoch']}",
        },
        {
            "title": "Stop when patience runs out",
            "explanation": f"If no improvement for {ex['patience']} consecutive epochs, stop training.",
            "math": r"\text{if wait} \geq \text{patience}: \text{stop}",
            "result": f"Stopped at epoch {ex['stop_epoch']}",
            "code": "if wait >= patience:\n    break",
        },
        {
            "title": "Why? Prevent overfitting",
            "explanation": "Training loss keeps decreasing but val loss starts increasing — the model is memorising the training data.",
            "answer": ex["answer"],
        },
    ]

def _early_stop_viz(ex):
    train_loss = ex["train_loss"]
    val_loss   = ex["val_loss"]
    stop_ep    = ex["stop_epoch"]
    best_ep    = ex["best_epoch"]
    epochs     = len(train_loss)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(epochs)
    ax.plot(x, train_loss, "#1D3557", lw=2, label="Training Loss")
    ax.plot(x, val_loss,   "#E63946", lw=2, label="Validation Loss")
    ax.axvline(best_ep,  color="#2A9D8F", lw=2, linestyle="--", label=f"Best val epoch ({best_ep})")
    ax.axvline(stop_ep,  color="black",   lw=2, linestyle=":",  label=f"Early stop epoch ({stop_ep})")
    ax.fill_between(x[best_ep:stop_ep+1], 0, max(train_loss) * 1.1,
                    alpha=0.08, color="#E63946", label=f"Patience window ({ex['patience']} epochs)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Early Stopping: Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Detect Overfitting / Underfitting
# ─────────────────────────────────────────────────────────────────────────────

def _overfit_gen(seed):
    rng = np.random.default_rng(seed)
    n   = 50
    ep  = np.arange(1, n + 1)

    # Underfitting: both losses high and plateaued
    underfit_train = 2.5 - 0.5 * np.exp(-0.05 * ep) + rng.normal(0, 0.03, n)
    underfit_val   = 2.6 - 0.4 * np.exp(-0.05 * ep) + rng.normal(0, 0.04, n)

    # Overfitting: train drops, val rises
    overfit_train  = 2.0 * np.exp(-0.1 * ep) + 0.1 + rng.normal(0, 0.02, n)
    overfit_val    = 2.0 * np.exp(-0.05 * ep) + 0.5 + 0.01 * ep + rng.normal(0, 0.04, n)

    # Good fit: both converge
    goodfit_train  = 2.0 * np.exp(-0.1 * ep) + 0.15 + rng.normal(0, 0.02, n)
    goodfit_val    = 2.0 * np.exp(-0.08 * ep) + 0.20 + rng.normal(0, 0.03, n)

    return {
        "inputs": {"epochs": n},
        "ep": ep,
        "underfit": (underfit_train, underfit_val),
        "overfit":  (overfit_train,  overfit_val),
        "goodfit":  (goodfit_train,  goodfit_val),
        "answer": "Compare train vs val loss curves to diagnose",
    }

def _overfit_steps(ex):
    return [
        {
            "title": "Underfitting: both train and val loss are high",
            "explanation": "Model is too simple or undertrained. Both losses plateau at a high value. Fix: increase model capacity or train longer.",
            "math": r"\mathcal{L}_{\text{train}} \approx \mathcal{L}_{\text{val}} \gg 0",
        },
        {
            "title": "Overfitting: train loss drops but val loss rises",
            "explanation": "Model memorises training data. Train loss ↓ but val loss ↑. Fix: regularisation, dropout, more data, early stopping.",
            "math": r"\mathcal{L}_{\text{train}} \ll \mathcal{L}_{\text{val}}",
        },
        {
            "title": "Good fit: both losses converge to a low value",
            "explanation": "Model generalises well. Train and val losses decrease together and converge. This is the goal.",
            "math": r"\mathcal{L}_{\text{train}} \approx \mathcal{L}_{\text{val}} \to \text{low}",
        },
    ]

def _overfit_viz(ex):
    ep = ex["ep"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cases = [
        (ex["underfit"], "Underfitting",  "#E9C46A"),
        (ex["overfit"],  "Overfitting",   "#E63946"),
        (ex["goodfit"],  "Good Fit",      "#2A9D8F"),
    ]
    for ax, ((train, val), title, color) in zip(axes, cases):
        ax.plot(ep, train, "#1D3557", lw=2, label="Train loss")
        ax.plot(ep, val,   color,     lw=2, label="Val loss",   linestyle="--")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(train.max(), val.max()) * 1.1)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

OPTIMIZATION_PROBLEMS = {
    "Gradient Descent": {
        "category": "Optimization",
        "difficulty": "Medium",
        "description": "Implement linear regression from scratch using gradient descent and watch the loss converge.",
        "tags": ["gradient descent", "optimization", "linear regression", "weights", "loss", "learning rate"],
        "generate": _gd_gen, "steps": _gd_steps, "visualize": _gd_viz,
    },
    "L2 Regularisation (Weight Decay)": {
        "category": "Optimization",
        "difficulty": "Medium",
        "description": "Add an L2 penalty to the loss function to shrink weights and prevent overfitting.",
        "tags": ["l2", "regularisation", "weight decay", "ridge", "lambda", "overfitting"],
        "generate": _l2_gen, "steps": _l2_steps, "visualize": _l2_viz,
    },
    "K-Fold Cross Validation": {
        "category": "Optimization",
        "difficulty": "Medium",
        "description": "Split data into K folds to get a reliable estimate of model generalisation.",
        "tags": ["cross validation", "kfold", "k-fold", "generalisation", "overfitting", "evaluation"],
        "generate": _kfold_gen, "steps": _kfold_steps, "visualize": _kfold_viz,
    },
    "Early Stopping": {
        "category": "Optimization",
        "difficulty": "Easy",
        "description": "Stop training when validation loss stops improving to prevent overfitting.",
        "tags": ["early stopping", "patience", "overfitting", "validation", "training"],
        "generate": _early_stop_gen, "steps": _early_stop_steps, "visualize": _early_stop_viz,
    },
    "Detect Overfitting / Underfitting": {
        "category": "Optimization",
        "difficulty": "Easy",
        "description": "Diagnose model behaviour by comparing training and validation learning curves.",
        "tags": ["overfitting", "underfitting", "learning curve", "generalisation", "bias variance"],
        "generate": _overfit_gen, "steps": _overfit_steps, "visualize": _overfit_viz,
    },
}
