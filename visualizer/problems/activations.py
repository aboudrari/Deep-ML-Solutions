import numpy as np
import matplotlib.pyplot as plt


def _activation_gen(seed, name):
    rng  = np.random.default_rng(seed)
    vals = np.round(rng.uniform(-3, 3, 6), 2)
    x    = np.linspace(-5, 5, 300)
    return vals, x


# ─────────────────────────────────────────────────────────────────────────────
# 1. ReLU & Leaky ReLU
# ─────────────────────────────────────────────────────────────────────────────

def _relu_gen(seed):
    vals, x = _activation_gen(seed, "relu")
    alpha = 0.1
    relu  = np.maximum(0, vals)
    lrelu = np.where(vals >= 0, vals, alpha * vals)
    return {
        "inputs": {"x values": vals, "alpha (Leaky ReLU)": alpha},
        "vals": vals, "alpha": alpha, "x": x,
        "relu": relu, "lrelu": lrelu,
        "answer": relu,
    }

def _relu_steps(ex):
    vals, alpha = ex["vals"], ex["alpha"]
    relu  = ex["relu"]
    lrelu = ex["lrelu"]
    return [
        {
            "title": "ReLU: clip negatives to zero",
            "explanation": "ReLU simply returns 0 for any negative input and x for positives.",
            "math": r"\text{ReLU}(x) = \max(0, x)",
            "result": relu,
            "code": "relu = np.maximum(0, x)",
        },
        {
            "title": "Why ReLU? — sparse activation",
            "explanation": "Negative neurons output exactly 0 (sparse), which is fast and reduces overfitting. But neurons can 'die' if all inputs are negative.",
        },
        {
            "title": "Leaky ReLU: allow small negative slope",
            "explanation": f"Instead of clipping to 0, multiply negatives by alpha={alpha}. Prevents dying neurons.",
            "math": r"\text{LeakyReLU}(x) = \begin{cases} x & x \geq 0 \\ \alpha x & x < 0 \end{cases}",
            "result": lrelu,
            "code": f"leaky = np.where(x >= 0, x, {alpha} * x)",
        },
        {
            "title": "ReLU derivative",
            "explanation": "Gradient is 1 for x>0 and 0 for x<0 (or α for Leaky ReLU). Simple and fast.",
            "math": r"\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}",
            "code": "grad = (x > 0).astype(float)",
        },
    ]

def _relu_viz(ex):
    x = ex["x"]
    alpha = ex["alpha"]
    relu   = np.maximum(0, x)
    lrelu  = np.where(x >= 0, x, alpha * x)
    d_relu = (x > 0).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(x, relu, "#E63946", lw=2.5)
    axes[0].scatter(ex["vals"], ex["relu"], color="#E63946", s=60, zorder=5, edgecolors="black")
    axes[0].axhline(0, color="gray", lw=0.5); axes[0].axvline(0, color="gray", lw=0.5)
    axes[0].set_title("ReLU: max(0, x)", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, relu,  "#E63946", lw=2, label="ReLU",         linestyle="--")
    axes[1].plot(x, lrelu, "#1D3557", lw=2.5, label=f"Leaky ReLU (α={alpha})")
    axes[1].axhline(0, color="gray", lw=0.5); axes[1].axvline(0, color="gray", lw=0.5)
    axes[1].set_title("ReLU vs Leaky ReLU", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, d_relu, "#2A9D8F", lw=2.5)
    axes[2].set_ylim(-0.2, 1.4)
    axes[2].axhline(0, color="gray", lw=0.5); axes[2].axvline(0, color="gray", lw=0.5)
    axes[2].set_title("ReLU Derivative", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sigmoid
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid_gen(seed):
    vals, x = _activation_gen(seed, "sigmoid")
    sig     = 1 / (1 + np.exp(-vals))
    return {
        "inputs": {"z values": vals},
        "vals": vals, "x": x, "sig": np.round(sig, 4),
        "answer": np.round(sig, 4),
    }

def _sigmoid_steps(ex):
    vals, sig = ex["vals"], ex["sig"]
    return [
        {
            "title": "Apply the sigmoid formula element-wise",
            "explanation": "Squashes any real number into the range (0, 1) — interpretable as a probability.",
            "math": r"\sigma(z) = \frac{1}{1 + e^{-z}}",
            "result": sig,
            "code": "sigmoid = 1 / (1 + np.exp(-z))",
        },
        {
            "title": "Understand saturation",
            "explanation": "For z ≫ 0 → σ ≈ 1, for z ≪ 0 → σ ≈ 0. Very large/small z causes vanishing gradients.",
            "math": r"\lim_{z \to +\infty} \sigma(z) = 1, \quad \lim_{z \to -\infty} \sigma(z) = 0",
        },
        {
            "title": "Sigmoid derivative (useful in backprop)",
            "explanation": "The derivative has a clean form in terms of the output itself.",
            "math": r"\sigma'(z) = \sigma(z)(1 - \sigma(z))",
            "code": "d_sigmoid = sigmoid * (1 - sigmoid)",
        },
    ]

def _sigmoid_viz(ex):
    x   = ex["x"]
    sig = 1 / (1 + np.exp(-x))
    d   = sig * (1 - sig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(x, sig, "#E63946", lw=2.5)
    ax1.axhline(0.5, color="gray", lw=1, linestyle="--", label="σ = 0.5 (threshold)")
    ax1.scatter(ex["vals"], ex["sig"], color="#1D3557", s=60, zorder=5, edgecolors="white", label="example inputs")
    ax1.set_title("Sigmoid σ(z) = 1/(1+e⁻ᶻ)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("z"); ax1.set_ylabel("σ(z)"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(x, d, "#2A9D8F", lw=2.5)
    ax2.set_title("Sigmoid Derivative  σ(z)(1−σ(z))", fontsize=12, fontweight="bold")
    ax2.set_xlabel("z"); ax2.set_ylabel("dσ/dz"); ax2.grid(True, alpha=0.3)
    ax2.annotate("Max gradient = 0.25\nat z = 0", xy=(0, 0.25), xytext=(1.5, 0.22),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Softmax
# ─────────────────────────────────────────────────────────────────────────────

def _softmax_gen(seed):
    rng    = np.random.default_rng(seed)
    logits = np.round(rng.uniform(-3, 4, 5), 2)
    e      = np.exp(logits - logits.max())   # numerically stable
    probs  = np.round(e / e.sum(), 6)
    return {
        "inputs": {"logits (raw scores)": logits},
        "logits": logits, "e": e, "probs": probs,
        "answer": probs,
    }

def _softmax_steps(ex):
    logits, e, probs = ex["logits"], ex["e"], ex["probs"]
    return [
        {
            "title": "Recall the Softmax formula",
            "explanation": "Converts a vector of raw scores (logits) into a probability distribution that sums to 1.",
            "math": r"\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}",
            "code": "probs = np.exp(z) / np.exp(z).sum()",
        },
        {
            "title": "Numerical stability trick: subtract the max",
            "explanation": "Subtracting max(z) before exponentiation avoids overflow without changing the result.",
            "math": r"\text{softmax}(z_i) = \frac{e^{z_i - z_{\max}}}{\sum_j e^{z_j - z_{\max}}}",
            "code": "z_stable = z - z.max()\nprobs = np.exp(z_stable) / np.exp(z_stable).sum()",
        },
        {
            "title": "Compute e^(z - max)",
            "explanation": f"logits - max = {np.round(logits - logits.max(), 4).tolist()}",
            "result": np.round(e, 4),
            "code": "e = np.exp(logits - logits.max())",
        },
        {
            "title": "Divide by the sum (normalise)",
            "explanation": f"Sum of e = {np.round(e.sum(), 4)}. Divide each element.",
            "result": probs,
            "code": "probs = e / e.sum()   # sums to 1.0",
        },
    ]

def _softmax_viz(ex):
    logits, probs = ex["logits"], ex["probs"]
    n = len(logits)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    labels = [f"Class {i}" for i in range(n)]
    ax1.bar(labels, logits, color="#1D3557", alpha=0.8, edgecolor="white")
    ax1.set_title("Raw Logits", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score")
    for i, v in enumerate(logits):
        ax1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    colors = plt.cm.RdYlGn(probs / probs.max())
    bars = ax2.bar(labels, probs, color=colors, edgecolor="white")
    ax2.set_title("Softmax Probabilities (sum = 1.0)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Probability"); ax2.set_ylim(0, min(1.0, probs.max() * 1.3))
    for bar, p in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width() / 2, p + 0.01, f"{p:.4f}",
                 ha="center", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tanh
# ─────────────────────────────────────────────────────────────────────────────

def _tanh_gen(seed):
    vals, x = _activation_gen(seed, "tanh")
    out  = np.round(np.tanh(vals), 4)
    return {
        "inputs": {"z values": vals},
        "vals": vals, "x": x, "out": out,
        "answer": out,
    }

def _tanh_steps(ex):
    vals, out = ex["vals"], ex["out"]
    return [
        {
            "title": "Apply the tanh formula",
            "explanation": "tanh maps inputs to (−1, 1) — zero-centered, unlike sigmoid.",
            "math": r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}",
            "result": out,
            "code": "out = np.tanh(z)",
        },
        {
            "title": "Relationship to sigmoid",
            "explanation": "tanh is a scaled & shifted sigmoid: tanh(z) = 2σ(2z) − 1",
            "math": r"\tanh(z) = 2\sigma(2z) - 1",
        },
        {
            "title": "tanh derivative",
            "explanation": "Max gradient = 1 at z=0 — stronger gradients than sigmoid (max 0.25).",
            "math": r"\tanh'(z) = 1 - \tanh^2(z)",
            "code": "d_tanh = 1 - np.tanh(z)**2",
        },
    ]

def _tanh_viz(ex):
    x = ex["x"]
    t = np.tanh(x)
    d = 1 - t**2
    s = 1 / (1 + np.exp(-x))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(x, t, "#E63946", lw=2.5)
    axes[0].scatter(ex["vals"], ex["out"], color="#1D3557", s=60, zorder=5, edgecolors="white")
    axes[0].axhline(0, color="gray", lw=0.5); axes[0].axvline(0, color="gray", lw=0.5)
    axes[0].set_title("tanh(z)", fontsize=12, fontweight="bold"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, t, "#E63946", lw=2, label="tanh  (range −1 to 1)")
    axes[1].plot(x, s, "#1D3557", lw=2, label="sigmoid  (range 0 to 1)", linestyle="--")
    axes[1].set_title("tanh vs Sigmoid", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, d, "#2A9D8F", lw=2.5)
    axes[2].set_title("tanh Derivative  1 − tanh²(z)", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Softsign
# ─────────────────────────────────────────────────────────────────────────────

def _softsign_gen(seed):
    vals, x = _activation_gen(seed, "softsign")
    out  = np.round(vals / (1 + np.abs(vals)), 4)
    return {
        "inputs": {"z values": vals},
        "vals": vals, "x": x, "out": out,
        "answer": out,
    }

def _softsign_steps(ex):
    vals, out = ex["vals"], ex["out"]
    return [
        {
            "title": "Apply Softsign",
            "explanation": "Like tanh but reaches saturation more slowly (polynomial, not exponential decay).",
            "math": r"\text{softsign}(z) = \frac{z}{1 + |z|}",
            "result": out,
            "code": "out = z / (1 + np.abs(z))",
        },
        {
            "title": "Softsign derivative",
            "explanation": "Gradient decays slower than tanh — can help in very deep networks.",
            "math": r"\frac{d}{dz}\text{softsign}(z) = \frac{1}{(1+|z|)^2}",
            "code": "d = 1 / (1 + np.abs(z))**2",
        },
    ]

def _softsign_viz(ex):
    x  = ex["x"]
    ss = x / (1 + np.abs(x))
    t  = np.tanh(x)
    d  = 1 / (1 + np.abs(x))**2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(x, ss, "#E63946", lw=2.5, label="Softsign")
    ax1.plot(x, t,  "#1D3557", lw=2,   label="tanh", linestyle="--")
    ax1.scatter(ex["vals"], ex["out"], color="#E63946", s=60, zorder=5, edgecolors="black")
    ax1.axhline(0, color="gray", lw=0.5); ax1.axvline(0, color="gray", lw=0.5)
    ax1.set_title("Softsign vs tanh", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(x, d, "#2A9D8F", lw=2.5)
    ax2.set_title("Softsign Derivative  1/(1+|z|)²", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Hard Sigmoid
# ─────────────────────────────────────────────────────────────────────────────

def _hardsig_gen(seed):
    vals, x = _activation_gen(seed, "hard_sigmoid")
    out  = np.round(np.clip((vals + 1) / 2, 0, 1), 4)
    return {
        "inputs": {"z values": vals},
        "vals": vals, "x": x, "out": out,
        "answer": out,
    }

def _hardsig_steps(ex):
    vals, out = ex["vals"], ex["out"]
    return [
        {
            "title": "Apply Hard Sigmoid",
            "explanation": "A piecewise linear approximation of sigmoid — much cheaper to compute.",
            "math": r"\text{HardSigmoid}(z) = \text{clip}\!\left(\frac{z+1}{2},\, 0,\, 1\right)",
            "result": out,
            "code": "out = np.clip((z + 1) / 2, 0, 1)",
        },
        {
            "title": "Understand the three regions",
            "explanation": "z ≤ −1 → output 0 | −1 < z < 1 → linear ramp | z ≥ 1 → output 1",
            "math": r"\text{HardSigmoid}(z) = \begin{cases} 0 & z \leq -1 \\ (z+1)/2 & -1 < z < 1 \\ 1 & z \geq 1 \end{cases}",
        },
        {
            "title": "Why use it?",
            "explanation": "Avoids exp() — useful on hardware with limited compute (mobile, embedded). Gradient is constant 0.5 in the active region.",
        },
    ]

def _hardsig_viz(ex):
    x   = ex["x"]
    hs  = np.clip((x + 1) / 2, 0, 1)
    sig = 1 / (1 + np.exp(-x))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, hs,  "#E63946", lw=2.5, label="Hard Sigmoid")
    ax.plot(x, sig, "#1D3557", lw=2, linestyle="--", label="Sigmoid (reference)")
    ax.scatter(ex["vals"], ex["out"], color="#E63946", s=60, zorder=5, edgecolors="black")
    ax.axvline(-1, color="gray", lw=0.8, linestyle=":")
    ax.axvline( 1, color="gray", lw=0.8, linestyle=":")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.fill_betweenx([0, 1], -1, 1, alpha=0.08, color="#2A9D8F", label="Linear region")
    ax.set_title("Hard Sigmoid vs Sigmoid", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ACTIVATION_PROBLEMS = {
    "ReLU & Leaky ReLU": {
        "category": "Activation Functions",
        "difficulty": "Easy",
        "description": "Compute ReLU and Leaky ReLU outputs and understand their gradients.",
        "tags": ["relu", "leaky relu", "activation", "neural network", "gradient"],
        "generate": _relu_gen, "steps": _relu_steps, "visualize": _relu_viz,
    },
    "Sigmoid": {
        "category": "Activation Functions",
        "difficulty": "Easy",
        "description": "Compute the sigmoid function and its derivative for a set of inputs.",
        "tags": ["sigmoid", "activation", "logistic", "probability", "binary"],
        "generate": _sigmoid_gen, "steps": _sigmoid_steps, "visualize": _sigmoid_viz,
    },
    "Softmax": {
        "category": "Activation Functions",
        "difficulty": "Easy",
        "description": "Convert a vector of raw logits into a probability distribution using Softmax.",
        "tags": ["softmax", "probability", "multi-class", "classification", "logits"],
        "generate": _softmax_gen, "steps": _softmax_steps, "visualize": _softmax_viz,
    },
    "Tanh": {
        "category": "Activation Functions",
        "difficulty": "Easy",
        "description": "Apply the hyperbolic tangent activation and compare with sigmoid.",
        "tags": ["tanh", "hyperbolic", "activation", "neural network", "zero centered"],
        "generate": _tanh_gen, "steps": _tanh_steps, "visualize": _tanh_viz,
    },
    "Softsign": {
        "category": "Activation Functions",
        "difficulty": "Easy",
        "description": "Compute the Softsign activation, which saturates more slowly than tanh.",
        "tags": ["softsign", "activation", "smooth", "neural network"],
        "generate": _softsign_gen, "steps": _softsign_steps, "visualize": _softsign_viz,
    },
    "Hard Sigmoid": {
        "category": "Activation Functions",
        "difficulty": "Easy",
        "description": "Compute the Hard Sigmoid — a piecewise linear approximation of sigmoid.",
        "tags": ["hard sigmoid", "activation", "piecewise", "efficient", "clip"],
        "generate": _hardsig_gen, "steps": _hardsig_steps, "visualize": _hardsig_viz,
    },
}
