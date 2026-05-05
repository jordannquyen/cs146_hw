import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})
OUTDIR = "homework2/"

df = pd.read_csv("homework2/heart_disease.csv")
print(f"\nDataset shape: {df.shape}")
print(df.describe())

fig, ax = plt.subplots(figsize=(7, 5))
colors = {0: '#2196F3', 1: '#F44336'}
labels_map = {0: 'No Heart Disease', 1: 'Heart Disease'}
for cls in [0, 1]:
    mask = df['target'] == cls
    ax.scatter(df.loc[mask, 'thalach'], df.loc[mask, 'oldpeak'],
               c=colors[cls], label=labels_map[cls], alpha=0.7, edgecolors='k', linewidths=0.3, s=50)
ax.set_xlabel('thalach (Max Heart Rate, bpm)')
ax.set_ylabel('oldpeak (ST Depression, mm)')
ax.set_title('Heart Disease Dataset – Raw Features')
ax.legend()
plt.tight_layout()
plt.savefig(OUTDIR + "p4a_scatter.png")
plt.close()

X = df[['thalach', 'oldpeak']].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n[4a] Train size: {X_train_s.shape[0]}, Test size: {X_test_s.shape[0]}")

def plot_decision_boundary(ax, clf, X_s, y, title, sv=None):
    h = 0.02
    x_min, x_max = X_s[:, 0].min() - 0.5, X_s[:, 0].max() + 0.5
    y_min, y_max = X_s[:, 1].min() - 0.5, X_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-3, 0, 3], colors=['#BBDEFB', '#FFCDD2'], alpha=0.5)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'],
               linestyles=['--', '-', '--'], linewidths=[1, 2, 1])

    for cls, c, lbl in [(0, '#2196F3', 'No HD'), (1, '#F44336', 'HD')]:
        mask = y == cls
        ax.scatter(X_s[mask, 0], X_s[mask, 1], c=c, label=lbl,
                   alpha=0.6, edgecolors='k', linewidths=0.3, s=30, zorder=3)
    if sv is not None:
        ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors='none',
                   edgecolors='gold', linewidths=2, zorder=4, label='SVs')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('thalach (std)')
    ax.set_ylabel('oldpeak (std)')

#b
svc_lin = SVC(kernel='linear', C=1.0)
svc_lin.fit(X_train_s, y_train)

train_acc_lin = accuracy_score(y_train, svc_lin.predict(X_train_s))
test_acc_lin  = accuracy_score(y_test,  svc_lin.predict(X_test_s))
print(f"\n[4b] Linear SVM  — Train acc: {train_acc_lin:.4f}  Test acc: {test_acc_lin:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
plot_decision_boundary(ax, svc_lin, X_train_s, y_train,
                       f'Linear SVM (C=1) — Test Acc={test_acc_lin:.3f}',
                       sv=svc_lin.support_vectors_)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR + "p4b_linear_svm.png")
plt.close()
print(f"   Support vectors: {svc_lin.support_vectors_.shape[0]}")

#4c
degrees = [2, 3, 5]
poly_svcs = {}
print("\n[4c] Polynomial SVMs:")
for d in degrees:
    clf = SVC(kernel='poly', degree=d, C=1.0)
    clf.fit(X_train_s, y_train)
    tr = accuracy_score(y_train, clf.predict(X_train_s))
    te = accuracy_score(y_test,  clf.predict(X_test_s))
    poly_svcs[d] = clf
    print(f"   degree={d}: Train={tr:.4f}  Test={te:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, d in zip(axes, degrees):
    clf = poly_svcs[d]
    te = accuracy_score(y_test, clf.predict(X_test_s))
    plot_decision_boundary(ax, clf, X_train_s, y_train,
                           f'Poly deg={d}  Test={te:.3f}')
plt.suptitle('Polynomial Kernel SVMs', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTDIR + "p4c_poly_svms.png", bbox_inches='tight')
plt.close()

#4d
gammas = [0.1, 1, 10, 100]
rbf_svcs = {}
print("\n[4d] RBF SVMs:")
for g in gammas:
    clf = SVC(kernel='rbf', gamma=g, C=1.0)
    clf.fit(X_train_s, y_train)
    tr = accuracy_score(y_train, clf.predict(X_train_s))
    te = accuracy_score(y_test,  clf.predict(X_test_s))
    rbf_svcs[g] = (clf, tr, te)
    print(f"   gamma={g:5.1f}: Train={tr:.4f}  Test={te:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, g in zip(axes.ravel(), gammas):
    clf, tr, te = rbf_svcs[g]
    plot_decision_boundary(ax, clf, X_train_s, y_train,
                           f'RBF γ={g}  Train={tr:.3f}  Test={te:.3f}')
plt.suptitle('RBF Kernel SVMs – varying γ', fontsize=13)
plt.tight_layout()
plt.savefig(OUTDIR + "p4d_rbf_svms.png", bbox_inches='tight')
plt.close()

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
tr_lr = accuracy_score(y_train, lr.predict(X_train_s))
te_lr = accuracy_score(y_test,  lr.predict(X_test_s))
print(f"\n[4e-i] Logistic Regression  Train={tr_lr:.4f}  Test={te_lr:.4f}")

h = 0.02
x_min, x_max = X_train_s[:, 0].min()-0.5, X_train_s[:, 0].max()+0.5
y_min, y_max = X_train_s[:, 1].min()-0.5, X_train_s[:, 1].max()+0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]

Z_svm = svc_lin.decision_function(grid).reshape(xx.shape)
Z_lr  = lr.decision_function(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 5))
for cls, c, lbl in [(0, '#2196F3', 'No HD'), (1, '#F44336', 'HD')]:
    mask = y_train == cls
    ax.scatter(X_train_s[mask, 0], X_train_s[mask, 1], c=c, label=lbl,
               alpha=0.5, s=30, edgecolors='k', linewidths=0.3)
ax.contour(xx, yy, Z_svm, levels=[0], colors=['black'],  linestyles=['-'],  linewidths=2)
ax.contour(xx, yy, Z_lr,  levels=[0], colors=['green'], linestyles=['--'], linewidths=2)
from matplotlib.lines import Line2D
handles_extra = [
    Line2D([0], [0], color='black',  lw=2, label='Linear SVM boundary'),
    Line2D([0], [0], color='green', lw=2, ls='--', label='Logistic Reg boundary'),
]
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(handles=handles+handles_extra, fontsize=8)
ax.set_title('Linear SVM vs Logistic Regression – Decision Boundaries')
ax.set_xlabel('thalach (std)')
ax.set_ylabel('oldpeak (std)')
plt.tight_layout()
plt.savefig(OUTDIR + "p4e_i_boundaries.png")
plt.close()

svc_rbf1 = rbf_svcs[1][0]
x_star = np.array([[10.0, 10.0]])
sv  = svc_rbf1.support_vectors_
dco = svc_rbf1.dual_coef_.ravel()
b   = svc_rbf1.intercept_[0]
gamma = 1.0

diff  = sv - x_star
k_vals = np.exp(-gamma * np.sum(diff**2, axis=1))
h_star = np.dot(dco, k_vals) + b

print(f"\n[4e-ii] RBF SVM (gamma=1) at x*=(10,10):")
print(f"   h(x*) = {h_star:.6f}")
print(f"   |h(x*) - b| = {abs(h_star - b):.2e}")