import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement

np.random.seed(42)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

df = pd.read_csv('question_4_data.csv')
feature_names = ['cement','slag','flyash','water','superplasticizer',
                 'coarseaggregate','fineaggregate','age']
target_name   = 'csMPa'

# (a)(i)
print("="*60)
print("PROBLEM 4(a)(i) — Basic Statistics")
print("="*60)
print(f"Shape: {df.shape}")
print(f"Samples: {df.shape[0]}, Features: {len(feature_names)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(df.dtypes)
print(df.describe())

# (a)(ii)
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(df[target_name], bins=30, color='steelblue', edgecolor='white')
ax.set_xlabel('Compressive Strength (MPa)')
ax.set_ylabel('Count')
ax.set_title('Target Distribution — Compressive Strength')
ax.axvline(df[target_name].mean(), color='red', linestyle='--', label=f"Mean={df[target_name].mean():.1f}")
ax.axvline(df[target_name].median(), color='orange', linestyle='--', label=f"Median={df[target_name].median():.1f}")
ax.legend()
plt.tight_layout()
plt.savefig('homework1-graphs/p4_a_ii_target_dist.png')
plt.close()
print(f"\n4(a)(ii) Target — Mean: {df[target_name].mean():.2f}, Std: {df[target_name].std():.2f}")
print("Distribution: right-skewed (long tail toward high strengths)")

# (a)(iii)
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, feat in enumerate(feature_names):
    axes[i].hist(df[feat], bins=30, color='teal', edgecolor='white')
    axes[i].set_title(feat)
    axes[i].set_xlabel('')
plt.suptitle('Feature Distributions', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('homework1-graphs/p4_a_iii_feature_dists.png', bbox_inches='tight')
plt.close()
print("\n4(a)(iii) Features with heavy zeros or skew:")
for feat in feature_names:
    zero_frac = (df[feat]==0).mean()
    if zero_frac > 0.1:
        print(f"  {feat}: {zero_frac*100:.1f}% zeros")

# (a)(iv)
corr = df.corr()
fig, ax = plt.subplots(figsize=(9,7))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = False
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax, annot_kws={'size':8})
ax.set_title('Pearson Correlation Matrix')
plt.tight_layout()
plt.savefig('homework1-graphs/p4_a_iv_corr.png')
plt.close()
target_corr = corr[target_name].drop(target_name).sort_values(key=abs, ascending=False)
print(f"\n4(a)(iv) Most positively correlated with strength: {target_corr.idxmax()} ({target_corr.max():.3f})")
print(f"Most negatively correlated: {target_corr.idxmin()} ({target_corr.min():.3f})")

# (a)(v)
top3 = target_corr.abs().nlargest(3).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, feat in zip(axes, top3):
    ax.scatter(df[feat], df[target_name], alpha=0.3, s=15, color='steelblue')
    ax.set_xlabel(feat)
    ax.set_ylabel('Compressive Strength (MPa)')
    ax.set_title(f'{feat} vs Strength (r={corr.loc[feat,target_name]:.2f})')
plt.suptitle('Top-3 Correlated Features vs Target', y=1.02)
plt.tight_layout()
plt.savefig('homework1-graphs/p4_a_v_scatter.png', bbox_inches='tight')
plt.close()
print(f"\n4(a)(v) Top-3 features by |correlation|: {top3}")

# ---

X_all = df[feature_names].values
y_all = df[target_name].values
N = len(y_all)

# (b)(i)
idx = np.random.permutation(N)
split = int(0.8 * N)
train_idx, test_idx = idx[:split], idx[split:]
X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
y_train, y_test         = y_all[train_idx], y_all[test_idx]
print(f"\n4(b)(i) Train: {len(y_train)} samples, Test: {len(y_test)} samples")

# (b)(iii)
mu  = X_train_raw.mean(axis=0)
sig = X_train_raw.std(axis=0)
X_train = (X_train_raw - mu) / sig
X_test  = (X_test_raw  - mu) / sig

# (b)(iv)
print("\n4(b)(iv) Train means ≈0:", np.round(X_train.mean(axis=0), 6))
print("Train stds  ≈1:", np.round(X_train.std(axis=0),  4))
print("Test means (≈0 but not exact):", np.round(X_test.mean(axis=0), 4))
print("Test stds  (≈1 but not exact):", np.round(X_test.std(axis=0),  4))

# --

def add_bias(X):
    return np.hstack([np.ones((X.shape[0],1)), X])

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

def predict(X, w):
    return X @ w

# -----

Xb_train = add_bias(X_train)
Xb_test  = add_bias(X_test)

w_lin = ols(Xb_train, y_train)
train_mse_lin = mse(predict(Xb_train, w_lin), y_train)
test_mse_lin  = mse(predict(Xb_test,  w_lin), y_test)

print(f"\n4(c)(i) Linear — Train MSE: {train_mse_lin:.2f}, Test MSE: {test_mse_lin:.2f}")

# (c)(ii)
abs_weights = np.abs(w_lin[1:])
top_feat_idx = np.argmax(abs_weights)
print(f"4(c)(ii) Largest weight: {feature_names[top_feat_idx]} (|w|={abs_weights[top_feat_idx]:.4f})")

from math import comb

def poly_features(X, degree):
    """All monomials up to given degree (no bias col)."""
    n, d = X.shape
    cols = [X]
    if degree >= 2:
        for i, j in combinations_with_replacement(range(d), 2):
            cols.append((X[:,i] * X[:,j]).reshape(-1,1))
    if degree >= 3:
        for i, j, k in combinations_with_replacement(range(d), 3):
            cols.append((X[:,i] * X[:,j] * X[:,k]).reshape(-1,1))
    return np.hstack(cols)

# (d)(i)
d = 8
n_deg2 = comb(d+2,2) - 1
n_deg3 = comb(d+3,3) - 1
print(f"\n4(d)(i) Degree-2 features (excl. bias): {n_deg2}  →  with bias: {n_deg2+1}")
print(f"        Degree-3 features (excl. bias): {n_deg3}  →  with bias: {n_deg3+1}")

# (d)(ii)

Xp2_train = add_bias(poly_features(X_train, 2))
Xp2_test  = add_bias(poly_features(X_test,  2))
w_p2 = ols(Xp2_train, y_train)
train_mse_p2 = mse(predict(Xp2_train, w_p2), y_train)
test_mse_p2  = mse(predict(Xp2_test,  w_p2), y_test)
print(f"\n4(d)(ii) Degree-2 — Train MSE: {train_mse_p2:.2f}, Test MSE: {test_mse_p2:.2f}")

# (d)(iii)

Xp3_train = add_bias(poly_features(X_train, 3))
Xp3_test  = add_bias(poly_features(X_test,  3))
w_p3 = ols(Xp3_train, y_train)
train_mse_p3 = mse(predict(Xp3_train, w_p3), y_train)
test_mse_p3  = mse(predict(Xp3_test,  w_p3), y_test)
print(f"4(d)(iii) Degree-3 — Train MSE: {train_mse_p3:.2f}, Test MSE: {test_mse_p3:.2f}")
print("  → Training MSE drops but test MSE may rise — classic overfitting signature")

#----------
lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

def ridge(X, y, lam):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)

ridge_train_mse, ridge_test_mse, ridge_w_norms = [], [], []
print("\n4(e)(i) Ridge on degree-3 features:")
print(f"{'lambda':>10} {'Train MSE':>12} {'Test MSE':>12}")
for lam in lambdas:
    w_r = ridge(Xp3_train, y_train, lam)
    tr  = mse(Xp3_train @ w_r, y_train)
    te  = mse(Xp3_test  @ w_r, y_test)
    ridge_train_mse.append(tr)
    ridge_test_mse.append(te)
    ridge_w_norms.append(np.linalg.norm(w_r)**2)
    print(f"{lam:>10.4f} {tr:>12.2f} {te:>12.2f}")

# (e)(ii)
fig, ax = plt.subplots(figsize=(7,4))
log_lam = np.log10(lambdas)
ax.plot(log_lam, ridge_train_mse, 'o-', label='Train MSE', color='steelblue')
ax.plot(log_lam, ridge_test_mse,  's-', label='Test MSE',  color='tomato')
ax.set_xlabel('log₁₀(λ)')
ax.set_ylabel('MSE')
ax.set_title('Ridge Regression: MSE vs Regularization Strength')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('homework1-graphs/p4_e_ii_ridge_mse.png')
plt.close()

# (e)(iii)
best_lam_idx = np.argmin(ridge_test_mse)
best_lam = lambdas[best_lam_idx]
print(f"\n4(e)(iii) Best λ by test MSE: {best_lam}  →  Test MSE: {ridge_test_mse[best_lam_idx]:.2f}")

# (e)(iv)
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(log_lam, ridge_w_norms, 'o-', color='purple')
ax.set_xlabel('log₁₀(λ)')
ax.set_ylabel('‖w‖²')
ax.set_title('Ridge: Weight Norm vs Regularization Strength')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('homework1-graphs/p4_e_iv_wnorm.png')
plt.close()

# ---------------------
K = 5
n_train = Xp3_train.shape[0]
fold_size = n_train // K
cv_idx = np.random.permutation(n_train)

print(f"\n4(f)(i) 5-Fold CV on degree-3 ridge:")
print(f"{'lambda':>10} {'Mean Val MSE':>15}")
cv_val_mses = []
for lam in lambdas:
    fold_mses = []
    for k in range(K):
        val_i   = cv_idx[k*fold_size:(k+1)*fold_size]
        train_i = np.concatenate([cv_idx[:k*fold_size], cv_idx[(k+1)*fold_size:]])
        Xtr, ytr = Xp3_train[train_i], y_train[train_i]
        Xval, yval = Xp3_train[val_i], y_train[val_i]
        w_cv = ridge(Xtr, ytr, lam)
        fold_mses.append(mse(Xval @ w_cv, yval))
    mean_val = np.mean(fold_mses)
    cv_val_mses.append(mean_val)
    print(f"{lam:>10.4f} {mean_val:>15.2f}")

best_cv_lam = lambdas[np.argmin(cv_val_mses)]
print(f"\n4(f)(ii) Best λ by CV: {best_cv_lam}")
w_final = ridge(Xp3_train, y_train, best_cv_lam)
final_test_mse = mse(Xp3_test @ w_final, y_test)
print(f"Final test MSE (CV-selected λ={best_cv_lam}): {final_test_mse:.2f}")

# (g)(i)
ages_train = X_train_raw[:, 7]  # raw ages
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(ages_train, bins=30, color='darkorange', edgecolor='white')
ax.set_xlabel('Age (days)')
ax.set_ylabel('Count')
ax.set_title('Age Feature Distribution (Training Set)')
plt.tight_layout()
plt.savefig('homework1-graphs/p4_g_i_age_hist.png')
plt.close()

age_vals = X_train_raw[:, 7]
unique_ages = np.unique(age_vals)
frac_28 = (age_vals == 28).mean()
print(f"\n4(g)(i) Distinct age values: {len(unique_ages)}")
print(f"Fraction with Age=28: {frac_28*100:.1f}%")

# (g)(ii)
age_counts = {a: (age_vals == a).sum() for a in unique_ages}
N_tr = len(y_train)
alpha = np.array([N_tr / age_counts[X_train_raw[i, 7]] for i in range(N_tr)])
A = np.diag(alpha)

Xb = Xb_train
w_wls = np.linalg.lstsq(Xb.T @ A @ Xb, Xb.T @ A @ y_train, rcond=None)[0]
train_mse_wls = mse(Xb_train @ w_wls, y_train)
test_mse_wls  = mse(Xb_test  @ w_wls, y_test)
print(f"4(g)(ii) WLS — Train MSE: {train_mse_wls:.2f}, Test MSE: {test_mse_wls:.2f}")
print(f"  vs OLS — Train MSE: {train_mse_lin:.2f}, Test MSE: {test_mse_lin:.2f}")

#--
print("\n" + "="*60)
print("4(h) MODEL SUMMARY TABLE")
print("="*60)
print(f"{'Model':<35} {'Train MSE':>12} {'Test MSE':>12}")
print("-"*60)
models = [
    ("Linear (degree-1)",           train_mse_lin,            test_mse_lin),
    ("Polynomial degree-2",         train_mse_p2,             test_mse_p2),
    ("Polynomial degree-3 (OLS)",   train_mse_p3,             test_mse_p3),
    (f"Ridge deg-3 λ={best_cv_lam} (CV)", mse(Xp3_train@w_final,y_train), final_test_mse),
    ("Weighted Linear (WLS)",       train_mse_wls,            test_mse_wls),
]
for name, tr, te in models:
    print(f"{name:<35} {tr:>12.2f} {te:>12.2f}")
