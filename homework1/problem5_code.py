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

df = pd.read_csv('train.csv')

print("="*60)
print("PROBLEM 5(a)(i) — First Look")
print("="*60)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.dtypes)
print(df.head())
numerical_cols   = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
categorical_cols = ['HomePlanet','CryoSleep','Destination','VIP']
print(f"\nNumerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

print("\n5(a)(ii) Missing Values:")
miss = pd.DataFrame({
    'count':   df.isnull().sum(),
    'percent': (df.isnull().mean()*100).round(2)
}).sort_values('percent', ascending=False)
print(miss[miss['count']>0])

# (a)(iii)
transported_frac = df['Transported'].mean()
print(f"\n5(a)(iii) Fraction Transported: {transported_frac:.3f} — dataset is ~balanced")

# (a)(iv)
fig, axes = plt.subplots(2, 3, figsize=(14,8))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col].dropna(), bins=40, color='steelblue', edgecolor='white')
    axes[i].set_title(col)
plt.suptitle('Numerical Feature Distributions', y=1.01)
plt.tight_layout()
plt.savefig('homework1-graphs/p5_a_iv_num_dists.png', bbox_inches='tight')
plt.close()
print("\n5(a)(iv) Right-skewed spending features:", [c for c in numerical_cols if c!='Age'])

# (a)(v)
fig, axes = plt.subplots(2, 2, figsize=(14,9))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    cats = df[col].dropna().unique()
    counts   = df[col].value_counts()
    trans_rate = df.groupby(col)['Transported'].mean()
    ax = axes[i]
    x = np.arange(len(counts))
    bars = ax.bar(x, counts.values, color='steelblue', alpha=0.7, label='Count')
    ax2 = ax.twinx()
    ax2.plot(x, [trans_rate.get(c, 0) for c in counts.index], 'ro-', label='Transport Rate')
    ax.set_xticks(x); ax.set_xticklabels(counts.index, rotation=30)
    ax.set_title(col); ax.set_ylabel('Count'); ax2.set_ylabel('Transport Rate')
    ax2.set_ylim(0,1)
plt.suptitle('Categorical Features: Count & Transport Rate', y=1.01)
plt.tight_layout()
plt.savefig('homework1-graphs/p5_a_v_cat_dists.png', bbox_inches='tight')
plt.close()
print("\n5(a)(v) Most predictive categorical: CryoSleep (large transport rate difference)")

# (a)(vi)
spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
fig, axes = plt.subplots(1, 5, figsize=(18,5))
for ax, col in zip(axes, spend_cols):
    groups = [df[df['Transported']==False][col].dropna(),
              df[df['Transported']==True][col].dropna()]
    ax.boxplot(groups, labels=['Not T','Transported'], showfliers=False)
    ax.set_title(col); ax.set_ylabel('Amount ($)')
plt.suptitle('Spending by Transported Status', y=1.02)
plt.tight_layout()
plt.savefig('homework1-graphs/p5_a_vi_spending.png', bbox_inches='tight')
plt.close()
print("5(a)(vi) Transported passengers tend to spend LESS (hypothesis: CryoSleep → can't spend)")

def preprocess(df_in, train_stats=None, is_train=True):
    df = df_in.copy()

    # (b)(i)
    df = df.drop(columns=['PassengerId','Name','Cabin'], errors='ignore')

    # (b)(ii)
    cat_cols = ['HomePlanet','Destination','CryoSleep','VIP']
    if is_train:
        modes = {c: df[c].mode()[0] for c in cat_cols}
    else:
        modes = train_stats['modes']
    for c in cat_cols:
        df[c] = df[c].fillna(modes[c])

    # (b)(iii)
    num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    if is_train:
        num_means = {c: df[c].mean() for c in num_cols}
    else:
        num_means = train_stats['num_means']
    for c in num_cols:
        df[c] = df[c].fillna(num_means[c])

# (b)(iv)
    df['CryoSleep'] = df['CryoSleep'].map({True:1, False:0, 'True':1, 'False':0}).astype(float)
    df['VIP']       = df['VIP'].map({True:1, False:0, 'True':1, 'False':0}).astype(float)
    if is_train:
        hp_cats  = sorted(df['HomePlanet'].unique())
        dst_cats = sorted(df['Destination'].unique())
    else:
        hp_cats  = train_stats['hp_cats']
        dst_cats = train_stats['dst_cats']

    for cat in hp_cats[1:]:
        df[f'HP_{cat}'] = (df['HomePlanet'] == cat).astype(float)
    for cat in dst_cats[1:]:
        df[f'Dst_{cat}'] = (df['Destination'] == cat).astype(float)
    df = df.drop(columns=['HomePlanet','Destination'])

    y = None
    if 'Transported' in df.columns:
        y = df['Transported'].map({True:1, False:0}).values.astype(float)
        df = df.drop(columns=['Transported'])

    X = df.values.astype(float)

    stats = None
    if is_train:
        stats = {'modes': modes, 'num_means': num_means,
                 'hp_cats': hp_cats, 'dst_cats': dst_cats}

    return X, y, stats

X_raw, y_all, train_stats = preprocess(df, is_train=True)

df_tmp = df.copy()
df_tmp = df_tmp.drop(columns=['PassengerId','Name','Cabin','Transported'], errors='ignore')
for c in ['HomePlanet','Destination','CryoSleep','VIP']:
    df_tmp[c] = df_tmp[c].fillna(df_tmp[c].mode()[0])
for c in ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']:
    df_tmp[c] = df_tmp[c].fillna(df_tmp[c].mean())
df_tmp['CryoSleep'] = df_tmp['CryoSleep'].map({True:1,False:0,'True':1,'False':0})
df_tmp['VIP']       = df_tmp['VIP'].map({True:1,False:0,'True':1,'False':0})
for cat in sorted(df_tmp['HomePlanet'].unique())[1:]:
    df_tmp[f'HP_{cat}'] = (df_tmp['HomePlanet']==cat).astype(float)
for cat in sorted(df_tmp['Destination'].unique())[1:]:
    df_tmp[f'Dst_{cat}'] = (df_tmp['Destination']==cat).astype(float)
df_tmp = df_tmp.drop(columns=['HomePlanet','Destination'])
feature_col_names = list(df_tmp.columns)

print(f"\n5(b)(iv) Total features after encoding: {X_raw.shape[1]}")
print("Any NaN in first 5 rows:", np.isnan(X_raw[:5]).any())
df_display = pd.DataFrame(X_raw[:5], columns=feature_col_names)
print("\nFirst 5 rows of processed feature matrix:")
print(df_display.to_string())

# (b)(v)
# 60 20 20
N = len(y_all)
idx = np.random.permutation(N)
n_test = int(0.2*N); n_val = int(0.2*N); n_train = N - n_test - n_val
train_idx = idx[:n_train]; val_idx = idx[n_train:n_train+n_val]; test_idx = idx[n_train+n_val:]
X_tr_raw, y_tr = X_raw[train_idx], y_all[train_idx]
X_val_raw, y_val = X_raw[val_idx],  y_all[val_idx]
X_te_raw, y_te  = X_raw[test_idx],  y_all[test_idx]
print(f"\n5(b)(v) Train: {len(y_tr)}, Val: {len(y_val)}, Test: {len(y_te)}")
print(f"  Class balance — Train: {y_tr.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_te.mean():.3f}")

# (b)(vi)
mu_tr  = X_tr_raw.mean(0); sig_tr = X_tr_raw.std(0)
sig_tr[sig_tr==0] = 1  # avoid div-by-zero for constant features
X_tr  = (X_tr_raw  - mu_tr) / sig_tr
X_val = (X_val_raw - mu_tr) / sig_tr
X_te  = (X_te_raw  - mu_tr) / sig_tr

def add_bias(X): return np.hstack([np.ones((len(X),1)), X])

Xb_tr  = add_bias(X_tr)
Xb_val = add_bias(X_val)
Xb_te  = add_bias(X_te)

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def ce_loss(w, X, y):
    p = sigmoid(X @ w)
    return -np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12))

def ce_grad(w, X, y):
    return X.T @ (sigmoid(X@w) - y) / len(y)

def mse_loss(w, X, y):
    p = sigmoid(X @ w)
    return np.mean((p - y)**2)

def mse_grad(w, X, y):
    p = sigmoid(X @ w)
    sp = p * (1 - p)
    return X.T @ (2*(p - y)*sp) / len(y)

def accuracy(w, X, y):
    return np.mean((sigmoid(X@w) >= 0.5) == y)

def gradient_descent(grad_fn, loss_fn, X, y, lr=0.1, iters=1000, lam=0):
    w = np.zeros(X.shape[1])
    losses = []
    for t in range(iters):
        g = grad_fn(w, X, y) + lam*w
        w -= lr * g
        losses.append(loss_fn(w, X, y))
    return w, losses

print("\n5(c) Training CE logistic regression...")
w_ce, losses_ce = gradient_descent(ce_grad, ce_loss, Xb_tr, y_tr, lr=0.1, iters=1000)

# (c)(ii
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(losses_ce, color='steelblue', linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('5(c)(ii) CE Loss vs Iteration (eta=0.1, 1000 iters)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('homework1-graphs/p5_c_ii_ce_loss.png')
plt.close()
print(f"  Initial CE loss: {losses_ce[0]:.4f}")
print(f"  Final CE loss:   {losses_ce[-1]:.4f}")
print(f"  Converged: {'Yes' if abs(losses_ce[-1]-losses_ce[-50]) < 1e-4 else 'Still decreasing slightly'}")
print(f"5(c)(iii) CE — Train acc: {accuracy(w_ce,Xb_tr,y_tr):.4f}, Val acc: {accuracy(w_ce,Xb_val,y_val):.4f}")

#---------------
print("\n5(d) Training MSE logistic regression...")
w_mse, losses_mse = gradient_descent(mse_grad, mse_loss, Xb_tr, y_tr, lr=0.1, iters=1000)
print(f"5(d)(iii) MSE — Train acc: {accuracy(w_mse,Xb_tr,y_tr):.4f}, Val acc: {accuracy(w_mse,Xb_val,y_val):.4f}")
fig, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(losses_ce, color='steelblue', label='CE Loss')
ax1.set_xlabel('Iteration'); ax1.set_ylabel('CE Loss', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax2 = ax1.twinx()
ax2.plot(losses_mse, color='tomato', linestyle='--', label='MSE Loss')
ax2.set_ylabel('MSE Loss', color='tomato')
ax2.tick_params(axis='y', labelcolor='tomato')
ax1.set_title('Training Loss: CE vs MSE (Logistic Regression)')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper right')
plt.tight_layout()
plt.savefig('homework1-graphs/p5_d_loss_curves.png')
plt.close()

def engineer_features(X_std, X_raw_orig):
    """Add zero-spend indicator and log(1+total_spend)."""
    # spending cols in raw are indices 1-5 (RoomService,FoodCourt,ShoppingMall,Spa,VRDeck)
    # after encoding we have: Age(0), RoomService(1)..VRDeck(5), CryoSleep(6), VIP(7), HP_*(8,9), Dst_*(10,11)
    spend_idx = [1,2,3,4,5]  # indices in original raw feature matrix
    total_spend = X_raw_orig[:, spend_idx].sum(axis=1)
    zero_ind    = (total_spend == 0).astype(float)
    log_spend   = np.log1p(total_spend)
    return zero_ind, log_spend, total_spend

zero_tr, log_tr, ts_tr   = engineer_features(X_tr,  X_tr_raw)
zero_val, log_val, ts_val = engineer_features(X_val, X_val_raw)
zero_te,  log_te,  ts_te  = engineer_features(X_te,  X_te_raw)

ls_mu, ls_sig = log_tr.mean(), log_tr.std()
log_tr_std  = (log_tr  - ls_mu) / ls_sig
log_val_std = (log_val - ls_mu) / ls_sig
log_te_std  = (log_te  - ls_mu) / ls_sig

X_eng_tr  = np.hstack([X_tr,  zero_tr[:,None],  log_tr_std[:,None]])
X_eng_val = np.hstack([X_val, zero_val[:,None], log_val_std[:,None]])
X_eng_te  = np.hstack([X_te,  zero_te[:,None],  log_te_std[:,None]])

Xb_eng_tr  = add_bias(X_eng_tr)
Xb_eng_val = add_bias(X_eng_val)
Xb_eng_te  = add_bias(X_eng_te)

print("\n5(e)(i) Training with engineered features...")
w_eng, _ = gradient_descent(ce_grad, ce_loss, Xb_eng_tr, y_tr, lr=0.1, iters=1000)
print(f"  Val acc with eng features: {accuracy(w_eng, Xb_eng_val, y_val):.4f}")

# (e)(ii
spend_std_tr  = X_tr[:,  [1,2,3,4,5]]
spend_std_val = X_val[:, [1,2,3,4,5]]
spend_std_te  = X_te[:,  [1,2,3,4,5]]

def spend_interactions(S):
    cols = []
    d = S.shape[1]
    for i in range(d):
        for j in range(i, d):
            cols.append((S[:,i]*S[:,j]).reshape(-1,1))
    return np.hstack(cols)

n_inter = len(list(combinations_with_replacement(range(5), 2)))
print(f"\n5(e)(ii) Interaction features added: {n_inter}")

inter_tr  = spend_interactions(spend_std_tr)
inter_val = spend_interactions(spend_std_val)
inter_te  = spend_interactions(spend_std_te)

X_poly_tr  = np.hstack([X_eng_tr,  inter_tr])
X_poly_val = np.hstack([X_eng_val, inter_val])
X_poly_te  = np.hstack([X_eng_te,  inter_te])

Xb_poly_tr  = add_bias(X_poly_tr)
Xb_poly_val = add_bias(X_poly_val)
Xb_poly_te  = add_bias(X_poly_te)

w_poly, _ = gradient_descent(ce_grad, ce_loss, Xb_poly_tr, y_tr, lr=0.1, iters=1000)
print(f"  Val acc with poly features: {accuracy(w_poly, Xb_poly_val, y_val):.4f}")
#-----
lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]

print("\n5(f)(ii) L2 Regularization sweep:")
print(f"{'lambda':>10} {'Train Acc':>12} {'Val Acc':>12}")
f_results = []
for lam in lambdas:
    w_r, _ = gradient_descent(ce_grad, ce_loss, Xb_poly_tr, y_tr, lr=0.1, iters=1000, lam=lam)
    tr_acc  = accuracy(w_r, Xb_poly_tr,  y_tr)
    val_acc = accuracy(w_r, Xb_poly_val, y_val)
    f_results.append((lam, tr_acc, val_acc, w_r))
    print(f"{lam:>10.4f} {tr_acc:>12.4f} {val_acc:>12.4f}")

best_f_lam_idx = np.argmax([r[2] for r in f_results])
best_f_lam = f_results[best_f_lam_idx][0]
print(f"\nBest λ (single split): {best_f_lam}")

# (f)(iii)
fig, ax = plt.subplots(figsize=(7,4))
log_lam = [np.log10(l) if l>0 else -5 for l in lambdas]
ax.plot(log_lam, [r[1] for r in f_results], 'o-', label='Train Acc', color='steelblue')
ax.plot(log_lam, [r[2] for r in f_results], 's-', label='Val Acc',   color='tomato')
ax.set_xlabel('log₁₀(λ)  (λ=0 shown at -5)')
ax.set_ylabel('Accuracy')
ax.set_title('L2 Regularization: Accuracy vs λ')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('homework1-graphs/p5_f_reg_acc.png')
plt.close()

K = 5
n_tr = len(y_tr)
fold_size = n_tr // K
cv_idx = np.random.permutation(n_tr)

print(f"\n5(g)(i) 5-Fold CV sweep:")
print(f"{'lambda':>10} {'Mean Val Acc':>15}")
cv_results = []
for lam in lambdas:
    fold_accs = []
    for k in range(K):
        val_i   = cv_idx[k*fold_size:(k+1)*fold_size]
        tr_i    = np.concatenate([cv_idx[:k*fold_size], cv_idx[(k+1)*fold_size:]])
        Xk, yk  = Xb_poly_tr[tr_i],  y_tr[tr_i]
        Xv, yv  = Xb_poly_tr[val_i], y_tr[val_i]
        w_k, _  = gradient_descent(ce_grad, ce_loss, Xk, yk, lr=0.1, iters=500, lam=lam)
        fold_accs.append(accuracy(w_k, Xv, yv))
    mean_acc = np.mean(fold_accs)
    cv_results.append(mean_acc)
    print(f"{lam:>10.4f} {mean_acc:>15.4f}")

best_cv_lam = lambdas[np.argmax(cv_results)]
print(f"\n5(g)(ii) Best λ by CV: {best_cv_lam}  |  Best λ by single split: {best_f_lam}")

# (g)(iii)
w_final, _ = gradient_descent(ce_grad, ce_loss, Xb_poly_tr, y_tr, lr=0.1, iters=1000, lam=best_cv_lam)
final_test_acc = accuracy(w_final, Xb_poly_te, y_te)
single_test_acc = accuracy(f_results[best_f_lam_idx][3], Xb_poly_te, y_te)
print(f"Final test acc (CV λ): {final_test_acc:.4f}")
print(f"Final test acc (single-split λ): {single_test_acc:.4f}")
# (h)(i)
val_acc_final = accuracy(w_final, Xb_poly_val, y_val)
print(f"\n5(h)(i) Best model — Val acc: {val_acc_final:.4f}, Test acc: {final_test_acc:.4f}")

# (h)(ii)
feature_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck',
                'CryoSleep','VIP'] + \
               [f'HP_{c}' for c in ['Europa','Mars']] + \
               [f'Dst_{c}' for c in ['PSO J318.5-22','TRAPPIST-1e']] + \
               ['ZeroSpend','LogSpend'] + \
               [f'Spend_inter_{i}' for i in range(n_inter)]
w_named = w_final[1:]  # skip bias
top3_idx = np.argsort(np.abs(w_named))[-3:][::-1]
print("\n5(h)(ii) Top-3 features by |weight|:")
for idx in top3_idx:
    name = feature_cols[idx] if idx < len(feature_cols) else f'feat_{idx}'
    print(f"  {name}: {w_named[idx]:.4f}")

# (h)(iii) 
# 6 12 index
feat_a_idx, feat_b_idx = 6, 12 
f_names = ['CryoSleep', 'ZeroSpend']

x_a = X_poly_te[:, feat_a_idx]
x_b = X_poly_te[:, feat_b_idx]

fig, ax = plt.subplots(figsize=(7,5))
colors = ['tomato' if yi==1 else 'steelblue' for yi in y_te]
ax.scatter(x_a, x_b, c=colors, alpha=0.3, s=15)
w0 = w_final[0]
wa = w_final[feat_a_idx+1]
wb = w_final[feat_b_idx+1]
x_range = np.linspace(x_a.min(), x_a.max(), 200)
if abs(wb) > 1e-6:
    x_b_line = -(w0 + wa*x_range) / wb
    ax.plot(x_range, x_b_line, 'k-', linewidth=2, label='Decision boundary')
ax.set_xlabel(f_names[0]); ax.set_ylabel(f_names[1])
ax.set_title('Decision Boundary: CryoSleep vs ZeroSpend')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='tomato',label='Transported'),
                   Patch(color='steelblue',label='Not Transported'),
                   plt.Line2D([0],[0],color='k',linewidth=2,label='Boundary')],
          loc='upper right')
plt.tight_layout()
plt.savefig('homework1-graphs/p5_h_iii_boundary.png')
plt.close()

# (h)(iv) summarize
print("\n" + "="*70)
print("5(h)(iv) MODEL SUMMARY TABLE")
print("="*70)
# recompute accuracies for all variants
w_base,   _ = gradient_descent(ce_grad, ce_loss, Xb_tr,      y_tr, lr=0.1, iters=1000)
w_mse2,   _ = gradient_descent(mse_grad, mse_loss, Xb_tr,   y_tr, lr=0.1, iters=1000)
w_eng2,   _ = gradient_descent(ce_grad, ce_loss, Xb_eng_tr,  y_tr, lr=0.1, iters=1000)

rows = [
    ("CE, base features",     Xb_tr, Xb_val, Xb_te, w_base),
    ("MSE, base features",    Xb_tr, Xb_val, Xb_te, w_mse2),
    ("CE, eng features",      Xb_eng_tr, Xb_eng_val, Xb_eng_te, w_eng2),
    ("CE, poly+eng, no reg",  Xb_poly_tr, Xb_poly_val, Xb_poly_te, f_results[0][3]),
    (f"CE, poly+eng, λ={best_cv_lam} (CV)", Xb_poly_tr, Xb_poly_val, Xb_poly_te, w_final),
]
print(f"{'Model':<40} {'#Feat':>6} {'Train':>8} {'Val':>8} {'Test':>8}")
print("-"*74)
for name, Xtr_m, Xval_m, Xte_m, w_m in rows:
    nf = Xtr_m.shape[1]-1
    tr  = accuracy(w_m, Xtr_m, y_tr)
    va  = accuracy(w_m, Xval_m, y_val)
    te  = accuracy(w_m, Xte_m,  y_te)
    print(f"{name:<40} {nf:>6} {tr:>8.4f} {va:>8.4f} {te:>8.4f}")