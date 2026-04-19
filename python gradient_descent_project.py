"""
Gradient Descent for Finding Minimum Error in AI Models
4th Semester Vector Calculus Project
Topic 1: Gradient Descent for Finding Minimum Error in AI Models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# 1. DATASET: House Price Prediction
# ─────────────────────────────────────────────
np.random.seed(42)
n = 50
area = np.random.uniform(500, 3000, n)           # sq ft
true_w = 0.15                                     # true weight
true_b = 50                                       # true bias
noise = np.random.normal(0, 20, n)
price = true_w * area + true_b + noise            # price in thousands

# Normalize features
X = (area - area.mean()) / area.std()
y = price

# ─────────────────────────────────────────────
# 2. VECTOR CALCULUS MODEL
#    y_pred = w*X + b
#    Loss  L = (1/2n) * Σ (y_pred - y)²
#    ∇_w L = (1/n) * Σ (y_pred - y) * X   ← partial derivative w.r.t. w
#    ∇_b L = (1/n) * Σ (y_pred - y)        ← partial derivative w.r.t. b
# ─────────────────────────────────────────────

def predict(X, w, b):
    return w * X + b

def mse_loss(y_pred, y):
    return np.mean((y_pred - y) ** 2) / 2

def gradient(X, y, w, b):
    n = len(y)
    y_pred = predict(X, w, b)
    error  = y_pred - y
    grad_w = np.mean(error * X)    # ∇_w L
    grad_b = np.mean(error)        # ∇_b L
    return grad_w, grad_b

def gradient_descent(X, y, lr=0.1, epochs=100):
    w, b = 0.0, 0.0
    history = {"loss": [], "w": [], "b": []}
    for _ in range(epochs):
        gw, gb = gradient(X, y, w, b)
        w -= lr * gw               # w ← w - α * ∇_w L
        b -= lr * gb               # b ← b - α * ∇_b L
        loss = mse_loss(predict(X, w, b), y)
        history["loss"].append(loss)
        history["w"].append(w)
        history["b"].append(b)
    return w, b, history

# Run with different learning rates
lrs      = [0.01, 0.05, 0.1, 0.3]
results  = {}
for lr in lrs:
    w, b, hist = gradient_descent(X, y, lr=lr, epochs=150)
    results[lr] = {"w": w, "b": b, "history": hist}

# Best model (lr=0.1)
best_w, best_b = results[0.1]["w"], results[0.1]["b"]

# ─────────────────────────────────────────────
# 3. LOSS SURFACE (for vector field / gradient vis)
# ─────────────────────────────────────────────
w_range = np.linspace(-5, 5, 60)
b_range = np.linspace(-5, 5, 60)
W, B    = np.meshgrid(w_range, b_range)
L       = np.array([[mse_loss(predict(X, wi, bi), y)
                     for wi in w_range] for bi in b_range])

# Gradient field (subsampled for quiver)
step = 5
dL_dw = np.gradient(L, axis=1)[::step, ::step]
dL_db = np.gradient(L, axis=0)[::step, ::step]

# ─────────────────────────────────────────────
# 4. FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ACCENT = "#00d4ff"
GOLD   = "#ffd700"
GREEN  = "#39ff14"
RED    = "#ff4c4c"
BG     = "#0d1117"
PANEL  = "#161b22"
TEXT   = "#e6edf3"

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title(title, color=ACCENT, fontsize=10, fontweight="bold", pad=8)

# ── Plot 1: Error vs Iterations (multiple LRs) ──
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1, "Loss vs Iterations — Different Learning Rates")
colors = [RED, GOLD, GREEN, ACCENT]
for lr, col in zip(lrs, colors):
    ax1.plot(results[lr]["history"]["loss"], color=col, lw=2, label=f"α = {lr}")
ax1.set_xlabel("Iteration", color=TEXT)
ax1.set_ylabel("MSE Loss", color=TEXT)
ax1.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, fontsize=9)
ax1.grid(alpha=0.15, color="#30363d")

# ── Plot 2: Data + Best Fit Line ──
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2, "House Price Prediction")
ax2.scatter(X, y, color=ACCENT, alpha=0.6, s=25, zorder=3)
x_line = np.linspace(X.min(), X.max(), 200)
ax2.plot(x_line, best_w * x_line + best_b, color=GREEN, lw=2.5, label="Fitted Line")
ax2.set_xlabel("Normalised Area", color=TEXT)
ax2.set_ylabel("Price (₹ thousands)", color=TEXT)
ax2.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, fontsize=9)
ax2.grid(alpha=0.15, color="#30363d")

# ── Plot 3: 3D Loss Surface ──
ax3 = fig.add_subplot(gs[1, :2], projection="3d")
ax3.set_facecolor(PANEL)
surf = ax3.plot_surface(W, B, L, cmap="plasma", alpha=0.85, linewidth=0)
fig.colorbar(surf, ax=ax3, shrink=0.5, pad=0.08, label="Loss")
# Plot gradient descent path
wh = results[0.1]["history"]["w"]
bh = results[0.1]["history"]["b"]
lh = results[0.1]["history"]["loss"]
ax3.plot(wh, bh, lh, color=GREEN, lw=2, zorder=10, label="GD Path (α=0.1)")
ax3.set_xlabel("w", color=TEXT, fontsize=8)
ax3.set_ylabel("b", color=TEXT, fontsize=8)
ax3.set_zlabel("Loss", color=TEXT, fontsize=8)
ax3.set_title("3D Loss Surface + Gradient Descent Path", color=ACCENT, fontsize=10, fontweight="bold")
ax3.tick_params(colors=TEXT, labelsize=7)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False

# ── Plot 4: Gradient (Vector) Field ──
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4, "Gradient Vector Field ∇L(w, b)")
cf = ax4.contourf(W, B, L, levels=20, cmap="plasma", alpha=0.7)
fig.colorbar(cf, ax=ax4, label="Loss")
ax4.quiver(W[::step, ::step], B[::step, ::step],
           -dL_dw, -dL_db,
           color="white", alpha=0.6, scale=40, width=0.004)
ax4.plot(wh, bh, color=GREEN, lw=2, label="GD Path")
ax4.scatter([wh[-1]], [bh[-1]], color=GREEN, s=60, zorder=5)
ax4.set_xlabel("w", color=TEXT)
ax4.set_ylabel("b", color=TEXT)
ax4.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, fontsize=9)

# ── Plot 5: Weight trajectory ──
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, "Weight (w) Convergence")
ax5.plot(results[0.1]["history"]["w"], color=GOLD, lw=2)
ax5.axhline(y=results[0.1]["w"], color=GREEN, lw=1.2, ls="--", label=f"Final w={best_w:.3f}")
ax5.set_xlabel("Iteration", color=TEXT)
ax5.set_ylabel("w value", color=TEXT)
ax5.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, fontsize=9)
ax5.grid(alpha=0.15, color="#30363d")

# ── Plot 6: Bias trajectory ──
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, "Bias (b) Convergence")
ax6.plot(results[0.1]["history"]["b"], color=RED, lw=2)
ax6.axhline(y=results[0.1]["b"], color=GREEN, lw=1.2, ls="--", label=f"Final b={best_b:.3f}")
ax6.set_xlabel("Iteration", color=TEXT)
ax6.set_ylabel("b value", color=TEXT)
ax6.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, fontsize=9)
ax6.grid(alpha=0.15, color="#30363d")

# ── Plot 7: Residuals ──
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7, "Residuals (Prediction Error)")
y_pred_final = predict(X, best_w, best_b)
residuals    = y - y_pred_final
ax7.scatter(y_pred_final, residuals, color=ACCENT, alpha=0.6, s=25)
ax7.axhline(0, color=GREEN, lw=1.5, ls="--")
ax7.set_xlabel("Predicted Price", color=TEXT)
ax7.set_ylabel("Residual", color=TEXT)
ax7.grid(alpha=0.15, color="#30363d")

fig.suptitle("Gradient Descent for Minimum Error — Vector Calculus Project",
             color=TEXT, fontsize=14, fontweight="bold", y=0.98)

plt.savefig("gradient_descent_visuals.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Plot saved.")

# ── Print final metrics ──
mse_final = mse_loss(y_pred_final, y) * 2
rmse      = np.sqrt(mse_final)
ss_res    = np.sum((y - y_pred_final) ** 2)
ss_tot    = np.sum((y - y.mean()) ** 2)
r2        = 1 - ss_res / ss_tot
print(f"Final w={best_w:.4f}, b={best_b:.4f}")
print(f"MSE={mse_final:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")