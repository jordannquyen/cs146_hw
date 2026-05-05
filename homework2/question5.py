import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 60)
print("PROBLEM 5: From-Scratch SVM – Pulsar Dataset")
print("=" * 60)
train_df = pd.read_csv("homework2/pulsar_data_train.csv")
test_df  = pd.read_csv("homework2/pulsar_data_test.csv")
OUTDIR = "homework2/"

feat_cols = [' Mean of the integrated profile',
             ' Standard deviation of the integrated profile',
             ' Excess kurtosis of the integrated profile',
             ' Skewness of the integrated profile',
             ' Mean of the DM-SNR curve',
             ' Standard deviation of the DM-SNR curve',
             ' Excess kurtosis of the DM-SNR curve',
             ' Skewness of the DM-SNR curve']
short_names = ['Mean_IP','SD_IP','EK_IP','Skew_IP','Mean_DM','SD_DM','EK_DM','Skew_DM']
label_col = 'target_class'

rename_map = {old: new for old, new in zip(feat_cols, short_names)}
rename_map[label_col] = 'label'
train_df = train_df.rename(columns=rename_map)
test_df  = test_df.rename(columns=rename_map)

#fill NaNs with column medians
medians = train_df[short_names].median()
train_df[short_names] = train_df[short_names].fillna(medians)
test_df[short_names]  = test_df[short_names].fillna(medians)

all_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"\n[5a] Full dataset shape: {all_df.shape}")
print(all_df[short_names + ['label']].describe().round(3).to_string())
pulsar_frac = all_df['label'].mean()
print(f"\nFraction of pulsars (class=1): {pulsar_frac:.4f}")
print(f"{1-pulsar_frac:.4f} accuracy")

# histogram
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, feat in zip(axes.ravel(), short_names):
    ax.hist(all_df.loc[all_df['label']==0, feat], bins=40, alpha=0.5,
            color='#2196F3', label='Non-pulsar', density=True)
    ax.hist(all_df.loc[all_df['label']==1, feat], bins=40, alpha=0.5,
            color='#F44336', label='Pulsar', density=True)
    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=7)
plt.suptitle('Feature Distributions by Class', fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR + "p5a_histograms.png", bbox_inches='tight')
plt.close()

# {0,1} -> {-1,+1}
def to_pm1(y): return 2*y - 1

X_tr_raw = train_df[short_names].values
y_tr      = to_pm1(train_df['label'].values)
X_te_raw  = test_df[short_names].values
y_te      = to_pm1(test_df["label"].fillna(0).values.astype(float))

X_tv, X_te2, y_tv, y_te2 = X_tr_raw, X_te_raw, y_tr, y_te
X_t60, X_val, y_t60, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25, random_state=42)

mu  = X_t60.mean(axis=0)
sig = X_t60.std(axis=0) + 1e-8
X_t60 = (X_t60 - mu) / sig
X_val = (X_val - mu) / sig
X_te2 = (X_te2 - mu) / sig

print(f"\n[5a] Train: {X_t60.shape}, Val: {X_val.shape}, Test: {X_te2.shape}")
N, D = X_t60.shape

def hinge_loss_and_grad(w, b, X, y, lam):
    margins = y * (X @ w + b) 
    mask    = margins < 1
    loss    = np.sum(np.maximum(0, 1 - margins)) + 0.5 * lam * np.dot(w, w)
    grad_w  = -((y[mask, None] * X[mask]).sum(axis=0)) + lam * w
    grad_b  = -(y[mask].sum())
    return loss, grad_w, grad_b

lam = 1e-3
eta = 1e-3
n_iters = 2000

np.random.seed(42)
w_h = np.zeros(D)
b_h = 0.0
losses_h = []

for t in range(n_iters):
    loss, gw, gb = hinge_loss_and_grad(w_h, b_h, X_t60, y_t60, lam)
    losses_h.append(loss)
    w_h -= eta * gw
    b_h -= eta * gb

print(f"\n[5b] Hinge SVM trained for {n_iters} iterations.")

def predict(w, b, X): return np.sign(X @ w + b)
tr_h  = accuracy_score(y_t60, predict(w_h, b_h, X_t60))
val_h = accuracy_score(y_val, predict(w_h, b_h, X_val))
print(f"   Train acc: {tr_h:.4f}   Val acc: {val_h:.4f}")

margins_tr = y_t60 * (X_t60 @ w_h + b_h)
inactive   = margins_tr >= 1
n_inactive = inactive.sum()
n_inactive_pos = ((y_t60[inactive]) == 1).sum()
n_inactive_neg = ((y_t60[inactive]) == -1).sum()
print(f"   Zero-subgradient examples (mn >= 1): {n_inactive} / {len(y_t60)}")
print(f"     Pulsars (class +1) inactive: {n_inactive_pos}")
print(f"     Non-pulsars (class -1) inactive: {n_inactive_neg}")

def sigmoid(z):
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def log_loss_and_grad(w, b, X, y, lam):
    margins = y * (X @ w + b)
    loss    = np.sum(np.logaddexp(0, -margins)) + 0.5 * lam * np.dot(w, w)
    s       = sigmoid(-margins)
    grad_w  = -(s * y) @ X + lam * w
    grad_b  = -(s * y).sum()
    return loss, grad_w, grad_b

w_l = np.zeros(D)
b_l = 0.0
losses_l = []

for t in range(n_iters):
    loss, gw, gb = log_loss_and_grad(w_l, b_l, X_t60, y_t60, lam)
    losses_l.append(loss)
    w_l -= eta * gw
    b_l -= eta * gb

print(f"\n[5c] Logistic SVM trained for {n_iters} iterations.")
tr_l  = accuracy_score(y_t60, predict(w_l, b_l, X_t60))
val_l = accuracy_score(y_val, predict(w_l, b_l, X_val))
print(f"   Train acc: {tr_l:.4f}   Val acc: {val_l:.4f}")

margins_tr_l = y_t60 * (X_t60 @ w_l + b_l)
s_vals       = sigmoid(-margins_tr_l)
grad_norms   = s_vals * np.linalg.norm(X_t60, axis=1)  # ||nabla_w ell_n||

n_exact_zero = (grad_norms == 0.0).sum()
n_below_1e6  = (grad_norms < 1e-6).sum()
print(f"   Exactly zero gradient norms: {n_exact_zero}")
print(f"   Gradient norms < 1e-6: {n_below_1e6}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(losses_h, label='Hinge loss', color='#2196F3')
ax.semilogy(losses_l, label='Logistic loss', color='#F44336')
ax.set_xlabel('Iteration')
ax.set_ylabel('Training Loss (log scale)')
ax.set_title('Training Loss Curves – Hinge vs Logistic')
ax.legend()
plt.tight_layout()
plt.savefig(OUTDIR + "p5bc_loss_curves.png")
plt.close()

margins_log = y_t60 * (X_t60 @ w_l + b_l)
s_log       = sigmoid(-margins_log)
x_norms     = np.linalg.norm(X_t60, axis=1)
grad_log    = s_log * x_norms
margins_hin = y_t60 * (X_t60 @ w_h + b_h)
grad_hin    = x_norms * (margins_hin < 1).astype(float)

fig, ax = plt.subplots(figsize=(9, 5))
idx = np.random.choice(len(margins_log), size=min(3000, len(margins_log)), replace=False)
ax.scatter(margins_log[idx], grad_log[idx], s=8, alpha=0.3, color='#F44336',
           label='Logistic gradient norm', zorder=2)
ax.scatter(margins_hin[idx], grad_hin[idx], s=8, alpha=0.3, color='#2196F3',
           label='Hinge subgradient norm', zorder=2)

m_grid = np.linspace(-4, 8, 400)
avg_xnorm = x_norms.mean()
ax.plot(m_grid, sigmoid(-m_grid) * avg_xnorm, 'r-', lw=2, label='σ(-m)·E[‖x‖] (theory)')
ax.axvline(x=1, color='gray', ls=':', lw=1.5, label='m=1 boundary (hinge kink)')
ax.set_xlabel('Margin  m = y(wᵀx + b)')
ax.set_ylabel('Per-example gradient norm  ‖∇_w ℓ_n‖')
ax.set_title('Margin vs Gradient Magnitude – Hinge vs Logistic')
ax.legend(fontsize=9)
ax.set_xlim(-4, 8); ax.set_ylim(-0.1, None)
plt.tight_layout()
plt.savefig(OUTDIR + "p5d_margin_gradient.png")
plt.close()

def train_model(X, y, loss='hinge', lam=1e-3, eta=1e-3, n_iters=2000):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(n_iters):
        if loss == 'hinge':
            _, gw, gb = hinge_loss_and_grad(w, b, X, y, lam)
        else:
            _, gw, gb = log_loss_and_grad(w, b, X, y, lam)
        w -= eta * gw
        b -= eta * gb
    return w, b

val_h_clean = accuracy_score(y_val, predict(w_h, b_h, X_val))
val_l_clean = accuracy_score(y_val, predict(w_l, b_l, X_val))
wnorm_h_clean = np.linalg.norm(w_h)
wnorm_l_clean = np.linalg.norm(w_l)

np.random.seed(0)
noise_mask = np.random.rand(len(y_t60)) < 0.05
y_noisy    = y_t60.copy()
y_noisy[noise_mask] *= -1

w_h_n, b_h_n = train_model(X_t60, y_noisy, 'hinge')
w_l_n, b_l_n = train_model(X_t60, y_noisy, 'logistic')
val_h_noisy = accuracy_score(y_val, predict(w_h_n, b_h_n, X_val))
val_l_noisy = accuracy_score(y_val, predict(w_l_n, b_l_n, X_val))
wnorm_h_noisy = np.linalg.norm(w_h_n)
wnorm_l_noisy = np.linalg.norm(w_l_n)

print(f"\n[5e] Label-noise robustness (5% flip):")
print(f"   Hinge:    Val acc clean={val_h_clean:.4f} -> noisy={val_h_noisy:.4f}  "
      f"Δ={val_h_noisy-val_h_clean:+.4f}")
print(f"   Logistic: Val acc clean={val_l_clean:.4f} -> noisy={val_l_noisy:.4f}  "
      f"Δ={val_l_noisy-val_l_clean:+.4f}")
print(f"   ||w|| Hinge:    clean={wnorm_h_clean:.4f} -> noisy={wnorm_h_noisy:.4f}  "
      f"Δ={wnorm_h_noisy-wnorm_h_clean:+.4f}")
print(f"   ||w|| Logistic: clean={wnorm_l_clean:.4f} -> noisy={wnorm_l_noisy:.4f}  "
      f"Δ={wnorm_l_noisy-wnorm_l_clean:+.4f}")

lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
tr_accs, val_accs = [], []

print("\n[5f] Regularisation sweep (logistic loss):")
print(f"   {'lambda':>10}  {'Train':>8}  {'Val':>8}")
for lam_sw in lambdas:
    w_sw, b_sw = train_model(X_t60, y_t60, 'logistic', lam=lam_sw)
    tr_sw  = accuracy_score(y_t60, predict(w_sw, b_sw, X_t60))
    val_sw = accuracy_score(y_val, predict(w_sw, b_sw, X_val))
    tr_accs.append(tr_sw)
    val_accs.append(val_sw)
    print(f"   {lam_sw:>10.4f}  {tr_sw:>8.4f}  {val_sw:>8.4f}")

best_idx = np.argmax(val_accs)
lam_star = lambdas[best_idx]
print(f"\n   Best lambda* = {lam_star} (val acc = {val_accs[best_idx]:.4f})")

fig, ax = plt.subplots(figsize=(7, 4))
log_lam = np.log10(lambdas)
ax.plot(log_lam, tr_accs,  'b-o', label='Train accuracy')
ax.plot(log_lam, val_accs, 'r-o', label='Val accuracy')
ax.axvline(np.log10(lam_star), color='green', ls='--', label=f'λ*={lam_star}')
ax.set_xlabel('log₁₀(λ)')
ax.set_ylabel('Accuracy')
ax.set_title('Regularisation Sweep – Logistic SVM')
ax.legend()
plt.tight_layout()
plt.savefig(OUTDIR + "p5f_reg_sweep.png")
plt.close()

X_tv_all = np.vstack([X_t60, X_val])
y_tv_all = np.concatenate([y_t60, y_val])
w_final, b_final = train_model(X_tv_all, y_tv_all, 'logistic', lam=lam_star)
test_final = accuracy_score(y_te2, predict(w_final, b_final, X_te2))
print(f"   Final test accuracy (retrained on train+val, λ*={lam_star}): {test_final:.4f}")

te_h  = accuracy_score(y_te2, predict(w_h, b_h, X_te2))
te_l  = accuracy_score(y_te2, predict(w_l, b_l, X_te2))
te_hn = accuracy_score(y_te2, predict(w_h_n, b_h_n, X_te2))
te_ln = accuracy_score(y_te2, predict(w_l_n, b_l_n, X_te2))

print("\n[5g] Summary Table")
print(f"{'Model':<30}  {'Train':>7}  {'Val':>7}  {'Test':>7}  {'||w||':>8}")
print("-" * 65)
rows = [
    ("Hinge SVM (clean)",          tr_h,       val_h_clean,  te_h,       wnorm_h_clean),
    ("Logistic SVM (clean)",       tr_l,       val_l_clean,  te_l,       wnorm_l_clean),
    ("Hinge SVM (5% noise)",       accuracy_score(y_t60, predict(w_h_n, b_h_n, X_t60)),
                                               val_h_noisy,  te_hn,      wnorm_h_noisy),
    ("Logistic SVM (5% noise)",    accuracy_score(y_t60, predict(w_l_n, b_l_n, X_t60)),
                                               val_l_noisy,  te_ln,      wnorm_l_noisy),
    (f"Logistic λ*={lam_star}",   val_accs[best_idx], val_accs[best_idx], test_final,
                                               np.linalg.norm(w_final)),
]
for name, tr, va, te, wn in rows:
    print(f"{name:<30}  {tr:>7.4f}  {va:>7.4f}  {te:>7.4f}  {wn:>8.4f}")