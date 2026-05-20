import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('homework3/figures', exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
PALETTE = sns.color_palette('tab10')
bank = pd.read_csv('homework3/bank-full.csv', sep=';')
print(f"Shape: {bank.shape}")
print(f"\ndtypes:\n{bank.dtypes}")
print(f"\nFirst 5 rows:\n{bank.head()}")

num_features = bank.select_dtypes(include=[np.number]).columns.tolist()
cat_features = bank.select_dtypes(include=['object']).columns.tolist()
print(f"\nNumerical features ({len(num_features)}): {num_features}")
print(f"Categorical features ({len(cat_features)}): {cat_features}")

yes_frac = (bank['y'] == 'yes').mean()
print(f"\nFraction 'yes': {yes_frac:.4f} ({yes_frac*100:.1f}%)")
print(f"Trivial 'always no' accuracy: {1-yes_frac:.4f}")
bank = bank.drop(columns=['duration'])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, feat in enumerate(['age', 'balance']):
    axes[i].hist(bank[bank['y'] == 'no'][feat],  bins=30, alpha=0.6,
                 label='no',  color=PALETTE[0])
    axes[i].hist(bank[bank['y'] == 'yes'][feat], bins=30, alpha=0.6,
                 label='yes', color=PALETTE[1])
    axes[i].set_xlabel(feat)
    axes[i].set_title(f'{feat} by subscription')
    axes[i].legend()

pout_rate = bank.groupby('poutcome')['y'].apply(lambda x: (x == 'yes').mean())
axes[2].bar(pout_rate.index, pout_rate.values, color=PALETTE[2])
axes[2].set_title('Subscription Rate by poutcome')
axes[2].set_ylabel('Subscription Rate')
axes[2].set_xlabel('poutcome')
plt.tight_layout()
plt.savefig('homework3/figures/p5a_eda.png')
plt.close()

bank['y'] = (bank['y'] == 'yes').astype(int)
cat_cols  = bank.select_dtypes(include='object').columns.tolist()
bank_enc  = pd.get_dummies(bank, columns=cat_cols, drop_first=True)
print(f"Features after one-hot encoding: {bank_enc.shape[1] - 1}")

X_bank = bank_enc.drop(columns=['y']).values
y_bank = bank_enc['y'].values
feature_names = bank_enc.drop(columns=['y']).columns.tolist()

np.random.seed(42)
idx = np.arange(len(X_bank))
np.random.shuffle(idx)
n_train   = int(0.6 * len(idx))
n_val     = int(0.2 * len(idx))
train_idx = idx[:n_train]
val_idx   = idx[n_train:n_train + n_val]
test_idx  = idx[n_train + n_val:]

X_tr, y_tr = X_bank[train_idx], y_bank[train_idx]
X_va, y_va = X_bank[val_idx],   y_bank[val_idx]
X_te, y_te = X_bank[test_idx],  y_bank[test_idx]
print(f"Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")
print(f"Class balance – Train: {y_tr.mean():.4f}, "
      f"Val: {y_va.mean():.4f}, Test: {y_te.mean():.4f}")

dt3 = DecisionTreeClassifier(max_depth=3, random_state=42)
dt3.fit(X_tr, y_tr)
print(f"Train accuracy: {dt3.score(X_tr, y_tr):.4f}")
print(f"Val   accuracy: {dt3.score(X_va, y_va):.4f}")

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt3, feature_names=feature_names, class_names=['no', 'yes'],
          filled=True, rounded=True, fontsize=7, ax=ax)
plt.title('Decision Tree (max_depth=3)')
plt.tight_layout()
plt.savefig('homework3/figures/p5c_tree_depth3.png')
plt.close()

tree = dt3.tree_
f0   = feature_names[tree.feature[0]]
f1   = feature_names[tree.feature[tree.children_left[0]]]
f2   = feature_names[tree.feature[tree.children_right[0]]]
print(f"\nTop 3 splits as if-then-else rules:")
print(f"  Root:        IF {f0} <= {tree.threshold[0]:.3f}")
print(f"  Left branch: IF {f1} <= {tree.threshold[tree.children_left[0]]:.3f}")
print(f"  Right branch:IF {f2} <= {tree.threshold[tree.children_right[0]]:.3f}")

depth_vals  = [1, 2, 3, 5, 7, 10, 15, 20, None]
train_accs  = []
val_accs    = []
for d in depth_vals:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_tr, y_tr)
    train_accs.append(dt.score(X_tr, y_tr))
    val_accs.append(dt.score(X_va, y_va))
    print(f"  depth={str(d):5s}: train={train_accs[-1]:.4f}, val={val_accs[-1]:.4f}")

depth_labels = [str(d) if d is not None else 'None' for d in depth_vals]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(depth_labels, train_accs, marker='o', label='Train',      color=PALETTE[0])
ax.plot(depth_labels, val_accs,   marker='s', label='Validation', color=PALETTE[1])
ax.set_xlabel('max_depth')
ax.set_ylabel('Accuracy')
ax.set_title('Decision Tree: Accuracy vs. Depth')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('homework3/figures/p5d_depth_accuracy.png')
plt.close()

best_val_idx = int(np.argmax(val_accs))
print(f"\nBest val depth={depth_labels[best_val_idx]}: {val_accs[best_val_idx]:.4f}")

X_tr_full = np.vstack([X_tr, X_va])
y_tr_full = np.concatenate([y_tr, y_va])
kf        = KFold(n_splits=5, shuffle=True, random_state=42)
cv_means, cv_stds = [], []

for d in depth_vals:
    scores = cross_val_score(
        DecisionTreeClassifier(max_depth=d, random_state=42),
        X_tr_full, y_tr_full, cv=kf)
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())
    print(f"  depth={str(d):5s}: CV mean={scores.mean():.4f} ± {scores.std():.4f}")

best_cv_idx = int(np.argmax(cv_means))
best_depth  = depth_vals[best_cv_idx]
print(f"\nBest CV depth: {best_depth}  (mean CV acc={cv_means[best_cv_idx]:.4f})")

dt_best      = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_best.fit(X_tr_full, y_tr_full)
test_acc_cv  = dt_best.score(X_te, y_te)
print(f"Test accuracy at best CV depth: {test_acc_cv:.4f}")

leaf_vals     = [1, 5, 20, 100, 500]
leaf_val_accs = []
for ml in leaf_vals:
    dt = DecisionTreeClassifier(max_depth=best_depth,
                                min_samples_leaf=ml, random_state=42)
    dt.fit(X_tr_full, y_tr_full)
    va_acc = dt.score(X_va, y_va)
    leaf_val_accs.append(va_acc)
    print(f"  min_samples_leaf={ml:4d}: val acc={va_acc:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(leaf_vals, leaf_val_accs, marker='o', color=PALETTE[3])
ax.set_xscale('log')
ax.set_xlabel('min_samples_leaf')
ax.set_ylabel('Validation Accuracy')
ax.set_title('Accuracy vs. min_samples_leaf')
plt.tight_layout()
plt.savefig('homework3/figures/p5f_leaf.png')
plt.close()

best_leaf_idx = int(np.argmax(leaf_val_accs))
best_leaf     = leaf_vals[best_leaf_idx]
dt_final      = DecisionTreeClassifier(max_depth=best_depth,
                                       min_samples_leaf=best_leaf,
                                       random_state=42)
dt_final.fit(X_tr_full, y_tr_full)
final_test_acc = dt_final.score(X_te, y_te)
print(f"\nBest min_samples_leaf={best_leaf}, test accuracy={final_test_acc:.4f}")

importances  = dt_final.feature_importances_
top15_idx    = np.argsort(importances)[::-1][:15]
top15_names  = [feature_names[i] for i in top15_idx]
top15_vals   = importances[top15_idx]

for name, val in zip(top15_names, top15_vals):
    print(f"  {name:40s}: {val:.5f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(15), top15_vals[::-1], color=PALETTE[4])
ax.set_yticks(range(15))
ax.set_yticklabels(top15_names[::-1], fontsize=8)
ax.set_xlabel('Feature Importance (Gini)')
ax.set_title('Top 15 Feature Importances')
plt.tight_layout()
plt.savefig('homework3/figures/p5g_importance.png')
plt.close()

#log reg
sc_bank  = StandardScaler()
X_tr_sc  = sc_bank.fit_transform(X_tr_full)
X_te_sc  = sc_bank.transform(X_te)
lr       = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_tr_sc, y_tr_full)
lr_acc   = lr.score(X_te_sc, y_te)
print(f"Logistic Regression test accuracy: {lr_acc:.4f}")

y_pred = dt_final.predict(X_te)
prec   = precision_score(y_te, y_pred, zero_division=0)
rec    = recall_score(y_te, y_pred,    zero_division=0)
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")