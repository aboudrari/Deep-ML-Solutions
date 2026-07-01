import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# 1. Orthogonal Projection
# ─────────────────────────────────────────────────────────────────────────────

def _proj_gen(seed):
    rng = np.random.default_rng(seed)
    u = np.round(rng.uniform(-3, 3, 2), 2)
    v = np.round(rng.uniform(-3, 3, 2), 2)
    while np.allclose(v, 0):
        v = np.round(rng.uniform(-3, 3, 2), 2)
    dot_uv = float(np.dot(u, v))
    dot_vv = float(np.dot(v, v))
    scalar  = dot_uv / dot_vv
    proj    = np.round(scalar * v, 6)
    return {
        "inputs": {"u (vector to project)": u, "v (target direction)": v},
        "u": u, "v": v,
        "dot_uv": dot_uv, "dot_vv": dot_vv,
        "scalar": scalar, "proj": proj,
        "answer": proj,
    }

def _proj_steps(ex):
    u, v = ex["u"], ex["v"]
    return [
        {
            "title": "Recall the formula",
            "explanation": "The projection of u onto v is a scaled version of v.",
            "math": r"\text{proj}_v(u) = \frac{u \cdot v}{v \cdot v} \cdot v",
            "code": "proj = (np.dot(u, v) / np.dot(v, v)) * v",
        },
        {
            "title": "Compute u · v (numerator)",
            "explanation": "Multiply element-wise and sum: " + " + ".join(f"({u[i]:.2f})×({v[i]:.2f})" for i in range(2)),
            "math": r"u \cdot v = " + " + ".join(f"({u[i]:.2f})({v[i]:.2f})" for i in range(2)) + f" = {ex['dot_uv']:.4f}",
            "result": ex["dot_uv"],
            "code": "dot_uv = np.dot(u, v)",
        },
        {
            "title": "Compute v · v (squared magnitude of v)",
            "explanation": "Square each element of v and sum.",
            "math": r"v \cdot v = " + " + ".join(f"({v[i]:.2f})^2" for i in range(2)) + f" = {ex['dot_vv']:.4f}",
            "result": ex["dot_vv"],
            "code": "dot_vv = np.dot(v, v)",
        },
        {
            "title": "Compute the scalar",
            "explanation": "Divide u·v by v·v to get the scaling factor.",
            "math": r"\text{scalar} = \frac{" + f"{ex['dot_uv']:.4f}" + r"}{" + f"{ex['dot_vv']:.4f}" + r"} = " + f"{ex['scalar']:.6f}",
            "result": ex["scalar"],
            "code": "scalar = dot_uv / dot_vv",
        },
        {
            "title": "Scale v by the scalar",
            "explanation": f"Multiply every element of v by {ex['scalar']:.4f}.",
            "math": r"\text{proj} = " + f"{ex['scalar']:.4f}" + r"\cdot" + str(v.tolist()),
            "result": ex["proj"],
            "code": "proj = scalar * v",
        },
    ]

def _proj_viz(ex):
    u, v, proj = ex["u"], ex["v"], ex["proj"]
    fig, ax = plt.subplots(figsize=(7, 6))
    O = np.zeros(2)

    def arrow(end, color, label):
        ax.annotate("", xy=end, xytext=O,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
        ax.text(end[0] * 1.1 + 0.1, end[1] * 1.1 + 0.1, label, color=color,
                fontsize=12, fontweight="bold")

    arrow(u,    "#E63946", "u")
    arrow(v,    "#1D3557", "v")
    arrow(proj, "#2A9D8F", f"proj\n{np.round(proj,2).tolist()}")
    ax.plot([u[0], proj[0]], [u[1], proj[1]], "k--", lw=1.2, alpha=0.5, label="perpendicular")

    all_pts = np.abs(np.concatenate([u, v, proj]))
    lim = max(all_pts) * 1.5 + 0.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Orthogonal Projection of u onto v", fontsize=13, fontweight="bold")
    patches = [
        mpatches.Patch(color="#E63946", label=f"u = {u.tolist()}"),
        mpatches.Patch(color="#1D3557", label=f"v = {v.tolist()}"),
        mpatches.Patch(color="#2A9D8F", label=f"proj = {np.round(proj,4).tolist()}"),
    ]
    ax.legend(handles=patches, fontsize=9)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Matrix Multiplication
# ─────────────────────────────────────────────────────────────────────────────

def _matmul_gen(seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, (3, 3)).astype(float)
    B = rng.integers(-4, 5, (3, 3)).astype(float)
    C = A @ B
    return {
        "inputs": {"A (3×3)": A, "B (3×3)": B},
        "A": A, "B": B, "C": C,
        "answer": C,
    }

def _matmul_steps(ex):
    A, B, C = ex["A"], ex["B"], ex["C"]
    return [
        {
            "title": "Recall the formula",
            "explanation": "Each element C[i,j] is the dot product of row i of A with column j of B.",
            "math": r"C_{ij} = \sum_{k} A_{ik} \cdot B_{kj}",
            "code": "C = A @ B",
        },
        {
            "title": "Compute C[0,0]",
            "explanation": f"Row 0 of A · Col 0 of B = {A[0].tolist()} · {B[:,0].tolist()}",
            "math": "C_{00} = " + " + ".join(f"({int(A[0,k])})({int(B[k,0])})" for k in range(3)) + f" = {int(C[0,0])}",
            "result": float(C[0, 0]),
            "code": "C[0,0] = np.dot(A[0], B[:, 0])",
        },
        {
            "title": "Compute C[0,1]",
            "explanation": f"Row 0 of A · Col 1 of B = {A[0].tolist()} · {B[:,1].tolist()}",
            "math": "C_{01} = " + " + ".join(f"({int(A[0,k])})({int(B[k,1])})" for k in range(3)) + f" = {int(C[0,1])}",
            "result": float(C[0, 1]),
            "code": "C[0,1] = np.dot(A[0], B[:, 1])",
        },
        {
            "title": "Complete all 9 elements",
            "explanation": "Repeat row×column dot products for all (i, j) combinations.",
            "result": C,
            "code": "C = A @ B  # numpy handles all elements at once",
        },
    ]

def _matmul_viz(ex):
    A, B, C = ex["A"], ex["B"], ex["C"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    def heatmap(ax, mat, title, cmap):
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="white" if abs(mat[i,j]) > 6 else "black")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    heatmap(axes[0], A, "Matrix A", "Blues")
    heatmap(axes[1], B, "Matrix B", "Greens")
    heatmap(axes[2], C, "C = A @ B", "Reds")
    fig.suptitle("Matrix Multiplication A @ B = C", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Eigenvalues & Eigenvectors
# ─────────────────────────────────────────────────────────────────────────────

def _eigen_gen(seed):
    rng = np.random.default_rng(seed)
    Q = rng.integers(-2, 3, (2, 2)).astype(float)
    A = np.round(Q @ Q.T + np.eye(2), 1)
    vals, vecs = np.linalg.eig(A)
    idx = np.argsort(vals)
    vals = np.real(vals[idx])
    vecs = np.real(vecs[:, idx])
    trace = float(np.trace(A))
    det   = float(np.linalg.det(A))
    disc  = trace ** 2 - 4 * det
    return {
        "inputs": {"A (2×2 matrix)": A},
        "A": A, "eigenvalues": vals, "eigenvectors": vecs,
        "trace": trace, "det": det, "disc": disc,
        "answer": np.round(vals, 4),
    }

def _eigen_steps(ex):
    A, trace, det, disc = ex["A"], ex["trace"], ex["det"], ex["disc"]
    lam1, lam2 = ex["eigenvalues"]
    return [
        {
            "title": "Set up the characteristic equation",
            "explanation": "Eigenvalues λ satisfy det(A − λI) = 0.",
            "math": r"\det(A - \lambda I) = 0",
            "code": "eigenvalues, eigenvectors = np.linalg.eig(A)",
        },
        {
            "title": "Expand the 2×2 determinant",
            "explanation": "For a 2×2 matrix the characteristic polynomial simplifies to:",
            "math": r"\lambda^2 - \mathrm{tr}(A)\,\lambda + \det(A) = 0",
        },
        {
            "title": "Compute trace and determinant",
            "explanation": f"tr(A) = {A[0,0]:.1f} + {A[1,1]:.1f} = {trace:.2f},   det(A) = {A[0,0]:.1f}×{A[1,1]:.1f} − ({A[0,1]:.1f})×({A[1,0]:.1f}) = {det:.2f}",
            "math": r"\lambda^2 - " + f"{trace:.2f}" + r"\lambda + " + f"{det:.2f}" + r" = 0",
            "result": f"trace = {trace:.2f},  det = {det:.2f}",
            "code": "trace = np.trace(A)\ndet   = np.linalg.det(A)",
        },
        {
            "title": "Solve with the quadratic formula",
            "explanation": f"Discriminant = {trace:.2f}² − 4×{det:.2f} = {disc:.2f}",
            "math": r"\lambda = \frac{\mathrm{tr}(A) \pm \sqrt{\mathrm{tr}(A)^2 - 4\det(A)}}{2}",
            "result": np.array([lam1, lam2]),
        },
        {
            "title": "Find eigenvectors (A − λI)v = 0",
            "explanation": "For each eigenvalue, solve the homogeneous system to get the eigenvector.",
            "math": r"(A - \lambda I)\,\mathbf{v} = \mathbf{0}",
            "result": ex["eigenvectors"],
            "code": "eigenvalues, eigenvectors = np.linalg.eig(A)",
        },
    ]

def _eigen_viz(ex):
    A, vals, vecs = ex["A"], ex["eigenvalues"], ex["eigenvectors"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#E63946", "#1D3557"]
    O = np.zeros(2)

    for i, (lam, color) in enumerate(zip(vals, colors)):
        v  = vecs[:, i] / (np.linalg.norm(vecs[:, i]) + 1e-9)
        Av = A @ v
        ax1.annotate("", xy=v,  xytext=O, arrowprops=dict(arrowstyle="-|>", color=color, lw=2, alpha=0.5))
        ax1.annotate("", xy=Av, xytext=O, arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, linestyle="dashed"))
        ax1.text(v[0] * 1.15, v[1] * 1.15, f"v{i+1}", color=color, fontsize=11)
        ax1.text(Av[0] * 1.1, Av[1] * 1.1, f"Av{i+1}=λ{i+1}v{i+1}\n(λ={lam:.2f})", color=color, fontsize=9)

    all_pts = np.abs(np.concatenate([vecs.ravel(), (A @ vecs).ravel()]))
    lim = max(all_pts) * 1.6 + 0.3
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.axhline(0, color="gray", lw=0.5); ax1.axvline(0, color="gray", lw=0.5)
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.3)
    ax1.set_title("Eigenvectors & Transformation Av = λv", fontsize=11, fontweight="bold")

    ax2.imshow(A, cmap="RdYlBu", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{A[i,j]:.1f}", ha="center", va="center", fontsize=16, fontweight="bold")
    ax2.set_title(f"Matrix A\nEigenvalues: {np.round(vals, 3).tolist()}", fontsize=12, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Jacobian Matrix
# ─────────────────────────────────────────────────────────────────────────────

def _jacobian_gen(seed):
    rng = np.random.default_rng(seed)
    x, y = np.round(rng.uniform(-2, 2, 2), 2)
    # f1 = x^2 * y,  f2 = 5x + sin(y)
    f1 = x**2 * y
    f2 = 5 * x + np.sin(y)
    # Partial derivatives
    df1_dx = 2 * x * y
    df1_dy = x**2
    df2_dx = 5.0
    df2_dy = np.cos(y)
    J = np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])
    return {
        "inputs": {"x": float(x), "y": float(y)},
        "x": float(x), "y": float(y),
        "f1": float(f1), "f2": float(f2),
        "J": J,
        "answer": np.round(J, 4),
    }

def _jacobian_steps(ex):
    x, y = ex["x"], ex["y"]
    J = ex["J"]
    return [
        {
            "title": "Define the vector-valued function",
            "explanation": "We have f: ℝ² → ℝ² defined as:",
            "math": r"f(x,y) = \begin{bmatrix} x^2 y \\ 5x + \sin(y) \end{bmatrix}",
            "code": "def f(x, y):\n    return [x**2 * y, 5*x + np.sin(y)]",
        },
        {
            "title": "Set up the Jacobian structure",
            "explanation": "The Jacobian is a matrix of all partial derivatives:",
            "math": r"J = \begin{bmatrix} \partial f_1/\partial x & \partial f_1/\partial y \\ \partial f_2/\partial x & \partial f_2/\partial y \end{bmatrix}",
        },
        {
            "title": "Compute row 1 partials (f₁ = x²y)",
            "explanation": f"∂f₁/∂x = 2xy = 2×{x:.2f}×{y:.2f} = {J[0,0]:.4f}   |   ∂f₁/∂y = x² = {x:.2f}² = {J[0,1]:.4f}",
            "math": r"\frac{\partial f_1}{\partial x} = 2xy, \quad \frac{\partial f_1}{\partial y} = x^2",
            "result": np.array([J[0, 0], J[0, 1]]),
            "code": "df1_dx = 2 * x * y\ndf1_dy = x**2",
        },
        {
            "title": "Compute row 2 partials (f₂ = 5x + sin(y))",
            "explanation": f"∂f₂/∂x = 5   |   ∂f₂/∂y = cos(y) = cos({y:.2f}) = {J[1,1]:.4f}",
            "math": r"\frac{\partial f_2}{\partial x} = 5, \quad \frac{\partial f_2}{\partial y} = \cos(y)",
            "result": np.array([J[1, 0], J[1, 1]]),
            "code": "df2_dx = 5\ndf2_dy = np.cos(y)",
        },
        {
            "title": "Assemble the Jacobian matrix",
            "explanation": f"Plug in x={x:.2f}, y={y:.2f}:",
            "math": r"J = \begin{bmatrix} 2xy & x^2 \\ 5 & \cos(y) \end{bmatrix}",
            "result": J,
            "code": "J = np.array([[2*x*y, x**2], [5, np.cos(y)]])",
        },
    ]

def _jacobian_viz(ex):
    J = ex["J"]
    x_range = np.linspace(-2, 2, 200)
    y_val   = ex["y"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: heatmap of Jacobian
    im = ax1.imshow(J, cmap="RdYlBu", aspect="auto")
    labels = [["∂f₁/∂x", "∂f₁/∂y"], ["∂f₂/∂x", "∂f₂/∂y"]]
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{labels[i][j]}\n{J[i,j]:.4f}", ha="center", va="center",
                     fontsize=12, fontweight="bold")
    ax1.set_title(f"Jacobian at x={ex['x']:.2f}, y={ex['y']:.2f}", fontsize=12, fontweight="bold")
    ax1.set_xticklabels([]); ax1.set_yticklabels([])
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["∂/∂x", "∂/∂y"]); ax1.set_yticklabels(["f₁", "f₂"])
    plt.colorbar(im, ax=ax1)

    # Right: plot both functions vs x (at fixed y)
    f1_vals = x_range**2 * y_val
    f2_vals = 5 * x_range + np.sin(y_val)
    ax2.plot(x_range, f1_vals, "#E63946", lw=2, label=f"f₁ = x²·{y_val:.2f}")
    ax2.plot(x_range, f2_vals, "#1D3557", lw=2, label=f"f₂ = 5x + sin({y_val:.2f})")
    ax2.axvline(ex["x"], color="gray", lw=1.5, linestyle="--", label=f"x = {ex['x']:.2f}")
    ax2.set_xlabel("x"); ax2.set_ylabel("f(x, y_fixed)")
    ax2.set_title(f"Functions at y = {y_val:.2f}", fontsize=12, fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Shift and Scale Array
# ─────────────────────────────────────────────────────────────────────────────

def _scale_gen(seed):
    rng  = np.random.default_rng(seed)
    arr  = np.round(rng.uniform(-10, 20, 6), 2)
    lo, hi = sorted(rng.integers(0, 5, 2).tolist())
    if lo == hi:
        hi = lo + 1
    arr_min, arr_max = arr.min(), arr.max()
    scaled = (arr - arr_min) / (arr_max - arr_min) * (hi - lo) + lo
    scaled = np.round(scaled, 4)
    return {
        "inputs": {"array": arr, "target min": float(lo), "target max": float(hi)},
        "arr": arr, "lo": float(lo), "hi": float(hi),
        "arr_min": float(arr_min), "arr_max": float(arr_max),
        "scaled": scaled,
        "answer": scaled,
    }

def _scale_steps(ex):
    arr, lo, hi = ex["arr"], ex["lo"], ex["hi"]
    return [
        {
            "title": "Recall the min-max scaling formula",
            "explanation": "Linearly maps any array into the target range [lo, hi].",
            "math": r"x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (\text{hi} - \text{lo}) + \text{lo}",
            "code": "scaled = (arr - arr.min()) / (arr.max() - arr.min()) * (hi - lo) + lo",
        },
        {
            "title": "Find min and max of the input array",
            "explanation": f"Scan the array: min = {ex['arr_min']:.2f}, max = {ex['arr_max']:.2f}",
            "result": f"min = {ex['arr_min']:.2f},  max = {ex['arr_max']:.2f}",
            "code": "arr_min = arr.min()\narr_max = arr.max()",
        },
        {
            "title": "Normalise to [0, 1]",
            "explanation": "Subtract min and divide by range so values land in [0, 1].",
            "math": r"x_{\text{norm}} = \frac{x - " + f"{ex['arr_min']:.2f}" + r"}{" + f"{ex['arr_max']:.2f}" + r" - " + f"{ex['arr_min']:.2f}" + r"}",
            "result": np.round((arr - ex["arr_min"]) / (ex["arr_max"] - ex["arr_min"]), 4),
            "code": "x_norm = (arr - arr_min) / (arr_max - arr_min)",
        },
        {
            "title": f"Stretch to [{lo:.0f}, {hi:.0f}]",
            "explanation": f"Multiply by (hi − lo) = {hi-lo:.0f} and add lo = {lo:.0f}.",
            "result": ex["scaled"],
            "code": f"scaled = x_norm * ({hi:.0f} - {lo:.0f}) + {lo:.0f}",
        },
    ]

def _scale_viz(ex):
    arr, scaled = ex["arr"], ex["scaled"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    idx = np.arange(len(arr))

    ax1.bar(idx, arr, color="#1D3557", alpha=0.8, edgecolor="white")
    ax1.set_title("Original Array", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Index"); ax1.set_ylabel("Value")
    for i, v in enumerate(arr):
        ax1.text(i, v + (0.3 if v >= 0 else -0.6), f"{v:.2f}", ha="center", fontsize=9)

    ax2.bar(idx, scaled, color="#2A9D8F", alpha=0.8, edgecolor="white")
    ax2.set_title(f"Scaled to [{ex['lo']:.0f}, {ex['hi']:.0f}]", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Index"); ax2.set_ylabel("Scaled Value")
    for i, v in enumerate(scaled):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. SVM Margin Width
# ─────────────────────────────────────────────────────────────────────────────

def _svm_gen(seed):
    rng = np.random.default_rng(seed)
    w   = np.round(rng.uniform(-3, 3, 2), 2)
    while np.allclose(w, 0):
        w = np.round(rng.uniform(-3, 3, 2), 2)
    b       = float(np.round(rng.uniform(-2, 2), 2))
    norm_w  = float(np.linalg.norm(w))
    margin  = round(2 / norm_w, 6)
    return {
        "inputs": {"w (weight vector)": w, "b (bias)": b},
        "w": w, "b": b,
        "norm_w": norm_w, "margin": margin,
        "answer": margin,
    }

def _svm_steps(ex):
    w = ex["w"]
    return [
        {
            "title": "Recall the SVM margin formula",
            "explanation": "The margin is the distance between the two support hyperplanes.",
            "math": r"\text{Margin} = \frac{2}{\|w\|}",
            "code": "margin = 2 / np.linalg.norm(w)",
        },
        {
            "title": "Compute ||w|| (L2 norm of weight vector)",
            "explanation": f"||w|| = sqrt({' + '.join(f'({wi:.2f})²' for wi in w)})",
            "math": r"\|w\| = \sqrt{" + " + ".join(f"({wi:.2f})^2" for wi in w) + r"} = " + f"{ex['norm_w']:.4f}",
            "result": ex["norm_w"],
            "code": "norm_w = np.linalg.norm(w)",
        },
        {
            "title": "Divide 2 by ||w||",
            "explanation": f"margin = 2 / {ex['norm_w']:.4f} = {ex['margin']:.6f}",
            "math": r"\text{Margin} = \frac{2}{" + f"{ex['norm_w']:.4f}" + r"} = " + f"{ex['margin']:.6f}",
            "result": ex["margin"],
            "code": "margin = 2 / norm_w",
        },
    ]

def _svm_viz(ex):
    w, b = ex["w"], ex["b"]
    norm_w = ex["norm_w"]
    fig, ax = plt.subplots(figsize=(7, 6))

    x_range = np.linspace(-4, 4, 200)

    if abs(w[1]) > 1e-6:
        # decision boundary: w·x + b = 0  →  x2 = (-w0*x1 - b) / w1
        y_db  = (-w[0] * x_range - b)         / w[1]
        y_pos = (-w[0] * x_range - b + 1)     / w[1]
        y_neg = (-w[0] * x_range - b - 1)     / w[1]
        ax.plot(x_range, y_db,  "k-",  lw=2,   label="Decision Boundary w·x+b=0")
        ax.plot(x_range, y_pos, "b--", lw=1.5, label="Positive margin w·x+b=+1")
        ax.plot(x_range, y_neg, "r--", lw=1.5, label="Negative margin w·x+b=−1")
        ax.fill_between(x_range, y_neg, y_pos, alpha=0.1, color="green",
                        label=f"Margin = {ex['margin']:.4f}")

    w_dir = w / norm_w
    ax.annotate("", xy=w_dir, xytext=np.zeros(2),
                arrowprops=dict(arrowstyle="-|>", color="purple", lw=2))
    ax.text(w_dir[0] * 1.1, w_dir[1] * 1.1, "w", color="purple", fontsize=13, fontweight="bold")

    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title(f"SVM Margin = 2/||w|| = {ex['margin']:.4f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

LINEAR_ALGEBRA_PROBLEMS = {
    "Orthogonal Projection": {
        "category": "Linear Algebra",
        "difficulty": "Medium",
        "description": "Project vector u onto vector v using the dot-product formula.",
        "tags": ["projection", "dot product", "vector", "linear algebra"],
        "generate": _proj_gen, "steps": _proj_steps, "visualize": _proj_viz,
    },
    "Matrix Multiplication": {
        "category": "Linear Algebra",
        "difficulty": "Easy",
        "description": "Multiply two matrices by computing dot products of rows and columns.",
        "tags": ["matrix", "matmul", "dot product", "linear algebra"],
        "generate": _matmul_gen, "steps": _matmul_steps, "visualize": _matmul_viz,
    },
    "Eigenvalues & Eigenvectors": {
        "category": "Linear Algebra",
        "difficulty": "Hard",
        "description": "Find eigenvalues and eigenvectors of a 2×2 matrix using the characteristic polynomial.",
        "tags": ["eigenvalue", "eigenvector", "characteristic polynomial", "PCA"],
        "generate": _eigen_gen, "steps": _eigen_steps, "visualize": _eigen_viz,
    },
    "Jacobian Matrix": {
        "category": "Linear Algebra",
        "difficulty": "Medium",
        "description": "Compute the matrix of all first-order partial derivatives of a vector-valued function.",
        "tags": ["jacobian", "partial derivative", "gradient", "calculus"],
        "generate": _jacobian_gen, "steps": _jacobian_steps, "visualize": _jacobian_viz,
    },
    "Shift and Scale Array": {
        "category": "Linear Algebra",
        "difficulty": "Easy",
        "description": "Linearly rescale an array to a target [min, max] range using min-max normalisation.",
        "tags": ["scaling", "normalisation", "min max", "preprocessing"],
        "generate": _scale_gen, "steps": _scale_steps, "visualize": _scale_viz,
    },
    "SVM Margin Width": {
        "category": "Linear Algebra",
        "difficulty": "Easy",
        "description": "Calculate the margin width of an SVM given the weight vector w.",
        "tags": ["svm", "margin", "support vector", "norm", "classification"],
        "generate": _svm_gen, "steps": _svm_steps, "visualize": _svm_viz,
    },
}
