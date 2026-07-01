import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ─────────────────────────────────────────────────────────────────────────────
# 1. F1 Score (Binary)
# ─────────────────────────────────────────────────────────────────────────────

def _f1_gen(seed):
    rng = np.random.default_rng(seed)
    n = 20
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "inputs": {"y_true": y_true, "y_pred": y_pred},
        "y_true": y_true, "y_pred": y_pred,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "answer": round(f1, 6),
    }

def _f1_steps(ex):
    tp, fp, fn, tn = ex["tp"], ex["fp"], ex["fn"], ex["tn"]
    p, r, f1 = ex["precision"], ex["recall"], ex["f1"]
    return [
        {
            "title": "Count TP, FP, FN, TN from predictions",
            "explanation": "Compare each prediction to the true label.",
            "math": r"\text{TP}=" + str(tp) + r",\ \text{FP}=" + str(fp) + r",\ \text{FN}=" + str(fn) + r",\ \text{TN}=" + str(tn),
            "result": f"TP={tp}, FP={fp}, FN={fn}, TN={tn}",
            "code": "tp = np.sum((y_true == 1) & (y_pred == 1))\nfp = np.sum((y_true == 0) & (y_pred == 1))\nfn = np.sum((y_true == 1) & (y_pred == 0))",
        },
        {
            "title": "Compute Precision",
            "explanation": f"Of all predicted positives, how many were correct? {tp} / ({tp}+{fp}) = {p:.4f}",
            "math": r"\text{Precision} = \frac{TP}{TP + FP} = \frac{" + str(tp) + r"}{" + str(tp + fp) + r"} = " + f"{p:.4f}",
            "result": p,
            "code": "precision = tp / (tp + fp)",
        },
        {
            "title": "Compute Recall",
            "explanation": f"Of all actual positives, how many did we catch? {tp} / ({tp}+{fn}) = {r:.4f}",
            "math": r"\text{Recall} = \frac{TP}{TP + FN} = \frac{" + str(tp) + r"}{" + str(tp + fn) + r"} = " + f"{r:.4f}",
            "result": r,
            "code": "recall = tp / (tp + fn)",
        },
        {
            "title": "Compute F1 Score (harmonic mean)",
            "explanation": f"Balances precision and recall. F1 = 2 × {p:.4f} × {r:.4f} / ({p:.4f} + {r:.4f}) = {f1:.4f}",
            "math": r"F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = " + f"{f1:.4f}",
            "result": f1,
            "code": "f1 = 2 * precision * recall / (precision + recall)",
        },
    ]

def _f1_viz(ex):
    tp, fp, fn, tn = ex["tp"], ex["fp"], ex["fn"], ex["tn"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    cm = np.array([[tn, fp], [fn, tp]])
    im = ax1.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Pred 0", "Pred 1"]); ax1.set_yticklabels(["True 0", "True 1"])
    ax1.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax1)

    metrics = ["Precision", "Recall", "F1"]
    values  = [ex["precision"], ex["recall"], ex["f1"]]
    colors  = ["#E63946", "#1D3557", "#2A9D8F"]
    bars = ax2.bar(metrics, values, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Metrics", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.3f}",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. KL Divergence (discrete)
# ─────────────────────────────────────────────────────────────────────────────

def _kl_gen(seed):
    rng = np.random.default_rng(seed)
    n = 5
    P = rng.dirichlet(np.ones(n))
    Q = rng.dirichlet(np.ones(n))
    P = np.round(P, 4); P = P / P.sum()
    Q = np.round(Q, 4); Q = Q / Q.sum()
    kl = float(np.sum(P * np.log(P / Q)))
    return {
        "inputs": {"P (true distribution)": P, "Q (approx distribution)": Q},
        "P": P, "Q": Q, "n": n, "kl": kl,
        "answer": round(kl, 6),
    }

def _kl_steps(ex):
    P, Q, kl = ex["P"], ex["Q"], ex["kl"]
    terms = P * np.log(P / Q)
    return [
        {
            "title": "Recall the KL Divergence formula",
            "explanation": "KL divergence measures how different Q is from P. It is NOT symmetric.",
            "math": r"D_{KL}(P \| Q) = \sum_{i} P(i) \log\frac{P(i)}{Q(i)}",
            "code": "kl = np.sum(P * np.log(P / Q))",
        },
        {
            "title": "Compute each term P(i) · log(P(i)/Q(i))",
            "explanation": "Compute log ratio and weight by P(i) for each category.",
            "result": np.round(terms, 6),
            "code": "terms = P * np.log(P / Q)",
        },
        {
            "title": "Sum all terms",
            "explanation": f"KL(P||Q) = {' + '.join(f'{t:.4f}' for t in terms)} = {kl:.6f}",
            "result": kl,
            "code": "kl = np.sum(terms)",
        },
    ]

def _kl_viz(ex):
    P, Q = ex["P"], ex["Q"]
    x = np.arange(len(P))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    w = 0.35
    ax1.bar(x - w/2, P, w, label="P (true)",  color="#E63946", alpha=0.8)
    ax1.bar(x + w/2, Q, w, label="Q (approx)", color="#1D3557", alpha=0.8)
    ax1.set_title(f"P vs Q\nKL(P||Q) = {ex['kl']:.4f}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Category"); ax1.set_ylabel("Probability")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    terms = P * np.log(P / Q)
    colors_t = ["#2A9D8F" if t >= 0 else "#E63946" for t in terms]
    ax2.bar(x, terms, color=colors_t, alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Contribution per category  P·log(P/Q)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Category"); ax2.set_ylabel("Contribution")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROC Curve & AUC
# ─────────────────────────────────────────────────────────────────────────────

def _auc_gen(seed):
    rng = np.random.default_rng(seed)
    n = 30
    y_true  = rng.integers(0, 2, n)
    scores  = np.clip(y_true * 0.6 + rng.uniform(-0.4, 0.4, n), 0, 1)
    scores  = np.round(scores, 3)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc_val = round(auc(fpr, tpr), 4)
    return {
        "inputs": {"y_true": y_true, "scores (predicted probabilities)": scores},
        "y_true": y_true, "scores": scores,
        "fpr": fpr, "tpr": tpr, "thresholds": thresholds,
        "auc": auc_val,
        "answer": auc_val,
    }

def _auc_steps(ex):
    return [
        {
            "title": "Sort predictions by score (descending)",
            "explanation": "Start with the highest threshold — at first only the most confident positives are selected.",
            "code": "sorted_idx = np.argsort(scores)[::-1]",
        },
        {
            "title": "Compute TPR and FPR at each threshold",
            "explanation": "Lower the threshold step by step. Each step adds one prediction as positive.",
            "math": r"\text{TPR} = \frac{TP}{TP+FN}, \quad \text{FPR} = \frac{FP}{FP+TN}",
            "code": "fpr, tpr, _ = roc_curve(y_true, scores)",
        },
        {
            "title": "Plot the ROC curve",
            "explanation": "The curve shows the trade-off between TPR (sensitivity) and FPR (1-specificity).",
        },
        {
            "title": "Compute AUC (area under the curve)",
            "explanation": "Use the trapezoidal rule. AUC = 0.5 is random, AUC = 1.0 is perfect.",
            "math": r"\text{AUC} = \int_0^1 \text{TPR}(\text{FPR})\,d\text{FPR}",
            "result": ex["auc"],
            "code": "auc_val = auc(fpr, tpr)",
        },
    ]

def _auc_viz(ex):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ex["fpr"], ex["tpr"], "#E63946", lw=2.5, label=f"ROC (AUC = {ex['auc']:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier (AUC = 0.5)")
    ax1.fill_between(ex["fpr"], ex["tpr"], alpha=0.15, color="#E63946")
    ax1.set_xlabel("FPR (False Positive Rate)"); ax1.set_ylabel("TPR (True Positive Rate)")
    ax1.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Score distribution by class
    pos_scores = ex["scores"][ex["y_true"] == 1]
    neg_scores = ex["scores"][ex["y_true"] == 0]
    ax2.hist(neg_scores, bins=10, alpha=0.7, color="#1D3557", label="Class 0")
    ax2.hist(pos_scores, bins=10, alpha=0.7, color="#E63946", label="Class 1")
    ax2.set_xlabel("Predicted Score"); ax2.set_ylabel("Count")
    ax2.set_title("Score Distribution by Class", fontsize=12, fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. MSE / MAE
# ─────────────────────────────────────────────────────────────────────────────

def _mse_gen(seed):
    rng   = np.random.default_rng(seed)
    n     = 8
    y_true = np.round(rng.uniform(0, 10, n), 2)
    noise  = np.round(rng.uniform(-2, 2, n), 2)
    y_pred = np.round(y_true + noise, 2)
    errors = y_pred - y_true
    mse  = float(np.mean(errors**2))
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    return {
        "inputs": {"y_true": y_true, "y_pred": y_pred},
        "y_true": y_true, "y_pred": y_pred,
        "errors": errors, "mse": mse, "mae": mae, "rmse": rmse,
        "answer": round(mse, 6),
    }

def _mse_steps(ex):
    n = len(ex["y_true"])
    errors = ex["errors"]
    return [
        {
            "title": "Compute errors (residuals)",
            "explanation": "Subtract true values from predictions.",
            "math": r"e_i = \hat{y}_i - y_i",
            "result": np.round(errors, 4),
            "code": "errors = y_pred - y_true",
        },
        {
            "title": "Square the errors",
            "explanation": "Squaring penalises large errors more than small ones.",
            "math": r"e_i^2",
            "result": np.round(errors**2, 4),
            "code": "sq_errors = errors**2",
        },
        {
            "title": "Compute MSE (mean of squared errors)",
            "explanation": f"Sum = {np.sum(errors**2):.4f} / {n} = {ex['mse']:.4f}",
            "math": r"\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}e_i^2 = " + f"{ex['mse']:.4f}",
            "result": ex["mse"],
            "code": "mse = np.mean(errors**2)",
        },
        {
            "title": "Compute MAE and RMSE",
            "explanation": "MAE uses absolute errors (less sensitive to outliers). RMSE = sqrt(MSE) in original units.",
            "math": r"\text{MAE} = " + f"{ex['mae']:.4f}" + r",\quad \text{RMSE} = " + f"{ex['rmse']:.4f}",
            "result": f"MAE={round(ex['mae'],4)},  RMSE={round(ex['rmse'],4)}",
            "code": "mae  = np.mean(np.abs(errors))\nrmse = np.sqrt(mse)",
        },
    ]

def _mse_viz(ex):
    y_true, y_pred = ex["y_true"], ex["y_pred"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    idx = np.arange(len(y_true))

    ax1.scatter(idx, y_true, color="#1D3557", s=80, zorder=5, label="y_true")
    ax1.scatter(idx, y_pred, color="#E63946", s=80, zorder=5, marker="x", linewidths=2, label="y_pred")
    for i in range(len(y_true)):
        ax1.plot([i, i], [y_true[i], y_pred[i]], "gray", lw=1.5, alpha=0.6)
    ax1.set_title(f"Predictions vs True\nMSE={ex['mse']:.3f}  MAE={ex['mae']:.3f}  RMSE={ex['rmse']:.3f}",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Sample index"); ax1.legend(); ax1.grid(True, alpha=0.3)

    errors = ex["errors"]
    colors = ["#E63946" if e > 0 else "#1D3557" for e in errors]
    ax2.bar(idx, errors, color=colors, alpha=0.8, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Residuals (y_pred − y_true)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sample index"); ax2.set_ylabel("Error")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Binary Cross-Entropy Loss
# ─────────────────────────────────────────────────────────────────────────────

def _bce_gen(seed):
    rng    = np.random.default_rng(seed)
    n      = 6
    y_true = rng.integers(0, 2, n)
    y_pred = np.round(np.clip(rng.uniform(0.05, 0.95, n), 0.01, 0.99), 3)
    terms  = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    loss   = float(np.mean(terms))
    return {
        "inputs": {"y_true": y_true, "y_pred (probabilities)": y_pred},
        "y_true": y_true, "y_pred": y_pred,
        "terms": terms, "loss": loss,
        "answer": round(loss, 6),
    }

def _bce_steps(ex):
    y_true, y_pred, terms = ex["y_true"], ex["y_pred"], ex["terms"]
    return [
        {
            "title": "Recall the Binary Cross-Entropy formula",
            "explanation": "Penalises confident wrong predictions exponentially.",
            "math": r"\mathcal{L} = -\frac{1}{n}\sum_{i}\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]",
            "code": "loss = -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))",
        },
        {
            "title": "Compute each term",
            "explanation": "For y=1: use −log(ŷ). For y=0: use −log(1−ŷ).",
            "result": np.round(terms, 4),
            "code": "terms = -(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))",
        },
        {
            "title": "Average the terms",
            "explanation": f"Sum = {np.sum(terms):.4f} / {len(terms)} = {ex['loss']:.6f}",
            "result": ex["loss"],
            "code": "loss = np.mean(terms)",
        },
    ]

def _bce_viz(ex):
    y_true, y_pred, terms = ex["y_true"], ex["y_pred"], ex["terms"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    p = np.linspace(0.01, 0.99, 200)
    ax1.plot(p, -np.log(p),     "#E63946", lw=2, label="−log(p)  [for y=1]")
    ax1.plot(p, -np.log(1 - p), "#1D3557", lw=2, label="−log(1−p)  [for y=0]")
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        color = "#E63946" if yt == 1 else "#1D3557"
        loss_i = -np.log(yp) if yt == 1 else -np.log(1 - yp)
        ax1.scatter(yp, loss_i, color=color, s=80, zorder=5)
    ax1.set_xlabel("Predicted probability ŷ"); ax1.set_ylabel("Loss")
    ax1.set_title("BCE Loss Curve", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)

    idx = np.arange(len(terms))
    colors = ["#E63946" if yt == 1 else "#1D3557" for yt in y_true]
    ax2.bar(idx, terms, color=colors, alpha=0.85, edgecolor="white")
    ax2.axhline(ex["loss"], color="black", lw=2, linestyle="--", label=f"Mean loss = {ex['loss']:.4f}")
    ax2.set_xlabel("Sample"); ax2.set_ylabel("Loss")
    ax2.set_title("Per-Sample Loss", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dice Score
# ─────────────────────────────────────────────────────────────────────────────

def _dice_gen(seed):
    rng    = np.random.default_rng(seed)
    n      = 15
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    inter  = int(np.sum(y_true & y_pred))
    union  = int(np.sum(y_true) + np.sum(y_pred))
    dice   = round(2 * inter / union if union > 0 else 0.0, 6)
    return {
        "inputs": {"y_true": y_true, "y_pred": y_pred},
        "y_true": y_true, "y_pred": y_pred,
        "inter": inter, "union_sum": union,
        "sum_true": int(np.sum(y_true)), "sum_pred": int(np.sum(y_pred)),
        "dice": dice,
        "answer": dice,
    }

def _dice_steps(ex):
    i, u = ex["inter"], ex["union_sum"]
    return [
        {
            "title": "Recall the Dice Score formula",
            "explanation": "Dice measures overlap between two binary sets.",
            "math": r"\text{Dice} = \frac{2 |A \cap B|}{|A| + |B|}",
            "code": "dice = 2 * np.sum(y_true & y_pred) / (np.sum(y_true) + np.sum(y_pred))",
        },
        {
            "title": "Compute intersection |A ∩ B|",
            "explanation": f"Count positions where both y_true and y_pred are 1: {i}",
            "result": i,
            "code": "inter = np.sum(y_true & y_pred)",
        },
        {
            "title": "Compute |A| + |B|",
            "explanation": f"|A| = {ex['sum_true']} (positives in y_true), |B| = {ex['sum_pred']} (positives in y_pred), sum = {u}",
            "result": u,
            "code": "denom = np.sum(y_true) + np.sum(y_pred)",
        },
        {
            "title": "Compute Dice Score",
            "explanation": f"Dice = 2 × {i} / {u} = {ex['dice']:.4f}",
            "math": r"\text{Dice} = \frac{2 \times " + str(i) + r"}{" + str(u) + r"} = " + f"{ex['dice']:.4f}",
            "result": ex["dice"],
        },
    ]

def _dice_viz(ex):
    y_true, y_pred = ex["y_true"], ex["y_pred"]
    fig, ax = plt.subplots(figsize=(10, 3))
    n = len(y_true)
    for i in range(n):
        a, b = bool(y_true[i]), bool(y_pred[i])
        if a and b:     c = "#2A9D8F"
        elif a and not b: c = "#E63946"
        elif b and not a: c = "#1D3557"
        else:             c = "#EEEEEE"
        ax.barh(0, 1, left=i, color=c, edgecolor="white", height=0.6)
        ax.barh(1, 1, left=i, color=c, edgecolor="white", height=0.6)

    # re-draw rows separately to show true/pred
    for i in range(n):
        a_c = "#E63946" if y_true[i] else "#EEEEEE"
        b_c = "#1D3557" if y_pred[i] else "#EEEEEE"
        both = bool(y_true[i]) and bool(y_pred[i])
        ax.barh(1, 1, left=i, color="#2A9D8F" if both else a_c, edgecolor="white", height=0.4)
        ax.barh(0, 1, left=i, color="#2A9D8F" if both else b_c, edgecolor="white", height=0.4)

    from matplotlib.patches import Patch
    patches = [
        Patch(color="#2A9D8F", label="Both 1 (Intersection)"),
        Patch(color="#E63946", label="Only y_true = 1"),
        Patch(color="#1D3557", label="Only y_pred = 1"),
        Patch(color="#EEEEEE", label="Both 0"),
    ]
    ax.set_yticks([0, 1]); ax.set_yticklabels(["y_pred", "y_true"])
    ax.set_xlabel("Sample index")
    ax.set_title(f"Dice Score = {ex['dice']:.4f}  |  Intersection = {ex['inter']}  |  |A|+|B| = {ex['union_sum']}",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

METRICS_PROBLEMS = {
    "F1 Score": {
        "category": "Metrics",
        "difficulty": "Easy",
        "description": "Compute the F1 score from binary predictions via precision and recall.",
        "tags": ["f1", "precision", "recall", "classification", "confusion matrix"],
        "generate": _f1_gen, "steps": _f1_steps, "visualize": _f1_viz,
    },
    "KL Divergence": {
        "category": "Metrics",
        "difficulty": "Medium",
        "description": "Measure how one probability distribution diverges from a reference distribution.",
        "tags": ["kl divergence", "entropy", "probability", "distribution", "information theory"],
        "generate": _kl_gen, "steps": _kl_steps, "visualize": _kl_viz,
    },
    "ROC Curve & AUC": {
        "category": "Metrics",
        "difficulty": "Medium",
        "description": "Plot the ROC curve and compute the Area Under the Curve (AUC) for a binary classifier.",
        "tags": ["roc", "auc", "tpr", "fpr", "threshold", "classification"],
        "generate": _auc_gen, "steps": _auc_steps, "visualize": _auc_viz,
    },
    "MSE / MAE / RMSE": {
        "category": "Metrics",
        "difficulty": "Easy",
        "description": "Compute Mean Squared Error, Mean Absolute Error, and Root MSE for regression predictions.",
        "tags": ["mse", "mae", "rmse", "regression", "loss", "error"],
        "generate": _mse_gen, "steps": _mse_steps, "visualize": _mse_viz,
    },
    "Binary Cross-Entropy": {
        "category": "Metrics",
        "difficulty": "Medium",
        "description": "Compute the binary cross-entropy (log loss) between true labels and predicted probabilities.",
        "tags": ["cross entropy", "log loss", "bce", "classification", "loss function"],
        "generate": _bce_gen, "steps": _bce_steps, "visualize": _bce_viz,
    },
    "Dice Score": {
        "category": "Metrics",
        "difficulty": "Easy",
        "description": "Compute the Dice coefficient measuring overlap between two binary prediction sets.",
        "tags": ["dice", "segmentation", "overlap", "f1", "binary"],
        "generate": _dice_gen, "steps": _dice_steps, "visualize": _dice_viz,
    },
}
