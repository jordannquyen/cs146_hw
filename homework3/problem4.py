import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('homework3/figures', exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import umap

plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
PALETTE = sns.color_palette('tab10')
X_gene = pd.read_csv('homework3/data.csv')
y_gene = pd.read_csv('homework3/labels.csv')
X_gene = X_gene.select_dtypes(include=[float, int]).reset_index(drop=True)
str_cols = y_gene.select_dtypes(include='object').columns.tolist()
if len(str_cols) == 0:
    pass
else:
    y_gene = y_gene[str_cols[-1]].rename('Class').to_frame()
if 'Class' not in y_gene.columns:
    y_gene.columns = ['Class']

print(f"X shape: {X_gene.shape}")
print(f"y shape: {y_gene.shape}")
print("\nSample counts per tumor type:\n", y_gene['Class'].value_counts())
sample_cols = X_gene.columns[:5].tolist()
print(f"\nDescribe (first 5 genes):\n", X_gene[sample_cols].describe().round(3))

X_np     = X_gene.values.astype(np.float64)
mu_gene  = X_np.mean(axis=0)
std_gene = X_np.std(axis=0)
std_gene[std_gene == 0] = 1
X_sc = (X_np - mu_gene) / std_gene
print(f"\nAfter standardization: mean≈{X_sc.mean():.4f}, std≈{X_sc.std():.4f}")
labels_gene = y_gene['Class'].values

def pca_svd(X, k):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components= Vt[:k]
    projected  = Xc @ components.T
    explained_var = S[:k]**2 / (X.shape[0] - 1)
    return components, projected, explained_var

components_svd, proj_svd, ev_svd = pca_svd(X_sc, k=2)
total_var = np.var(X_sc, axis=0, ddof=1).sum()
print(f"Explained variance – PC1: {ev_svd[0]:.4f},  PC2: {ev_svd[1]:.4f}")
print(f"Fraction explained – PC1: {ev_svd[0]/total_var*100:.2f}%,  "
      f"PC2: {ev_svd[1]/total_var*100:.2f}%")

def pca_eig(X, k):
    Xc = X - X.mean(axis=0)
    N  = Xc.shape[0]
    G  = Xc @ Xc.T / (N - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx  = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors= eigenvectors[:, idx]
    components = np.array([
        Xc.T @ eigenvectors[:, j] / np.sqrt((N - 1) * max(eigenvalues[j], 1e-12))
        for j in range(k)
    ])
    projected= Xc @ components.T
    explained_var = eigenvalues[:k]
    return components, projected, explained_var

components_eig, proj_eig, ev_eig = pca_eig(X_sc, k=2)
print(f"SVD explained variances: {ev_svd.round(4)}")
print(f"EIG explained variances: {ev_eig.round(4)}")
print(f"|SVD - EIG| difference:  {np.abs(ev_svd - ev_eig).round(8)}")
for i in range(2):
    print(f"PC{i+1} |cos(svd, eig)|: {abs(np.dot(components_svd[i], components_eig[i])):.8f}")

_, _, ev_100   = pca_svd(X_sc, k=100)
total_var_gene = np.var(X_sc, axis=0, ddof=1).sum()
frac_explained = np.cumsum(ev_100) / total_var_gene

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, 101), ev_100, color=PALETTE[0])
axes[0].set_xlabel('Component')
axes[0].set_ylabel('Explained Variance')
axes[0].set_title('Scree Plot')
axes[1].plot(range(1, 101), frac_explained * 100, color=PALETTE[1])
axes[1].axhline(80, color='red', linestyle='--', label='80%')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
plt.tight_layout()
plt.savefig('homework3/figures/p4d_scree.png')
plt.close()

n_80 = int(np.argmax(frac_explained >= 0.80)) + 1
print(f"Components needed for 80% variance: {n_80} (out of 20,531 genes)")

class_colors = dict(zip(['BRCA','KIRC','COAD','LUAD','PRAD'], PALETTE[:5]))
fig, ax = plt.subplots(figsize=(8, 6))
for cls in np.unique(labels_gene):
    mask = labels_gene == cls
    ax.scatter(proj_svd[mask, 0], proj_svd[mask, 1], c=[class_colors[cls]], label=cls, alpha=0.6, s=20)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('2D PCA – Gene Expression (801 Tumors)')
ax.legend()
plt.tight_layout()
plt.savefig('homework3/figures/p4e_pca2d.png')
plt.close()

gene_names    = X_gene.columns.tolist()
pc1 = components_svd[0]
pc2= components_svd[1]
top10_pc1_idx = np.argsort(np.abs(pc1))[::-1][:10]
top10_pc2_idx = np.argsort(np.abs(pc2))[::-1][:10]

print("Top 10 genes by |loading| on PC1:")
for i, idx in enumerate(top10_pc1_idx):
    print(f"  {i+1}. {gene_names[idx]}: loading = {pc1[idx]:.5f}")
print("\nTop 10 genes by |loading| on PC2:")
for i, idx in enumerate(top10_pc2_idx):
    print(f"  {i+1}. {gene_names[idx]}: loading = {pc2[idx]:.5f}")

reducer   = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(X_sc)
print(f"UMAP embedding shape: {embedding.shape}")

fig, ax = plt.subplots(figsize=(8, 6))
for cls in np.unique(labels_gene):
    mask = labels_gene == cls
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               c=[class_colors[cls]], label=cls, alpha=0.6, s=20)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.set_title('UMAP Embedding – Gene Expression')
ax.legend()
plt.tight_layout()
plt.savefig('homework3/figures/p4g_umap.png')
plt.close()

n_neighbors_vals = [5, 15, 50]
min_dist_vals= [0.0, 0.5]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row_i, min_dist in enumerate(min_dist_vals):
    for col_i, nn in enumerate(n_neighbors_vals):
        ax  = axes[row_i][col_i]
        emb = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=min_dist,
                        random_state=42).fit_transform(X_sc)
        for cls in np.unique(labels_gene):
            mask = labels_gene == cls
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[class_colors[cls]],
                       label=cls, alpha=0.5, s=8)
        ax.set_title(f'n_neighbors={nn}, min_dist={min_dist}')
        ax.set_xticks([]); ax.set_yticks([])
handles, labs = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labs, loc='lower center', ncol=5, fontsize=9)
plt.suptitle('UMAP Hyperparameter Grid', fontsize=13)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('homework3/figures/p4h_umap_grid.png')
plt.close()

le    = LabelEncoder()
y_enc = le.fit_transform(labels_gene)

Xc_all  = X_sc - X_sc.mean(axis=0)
_, _, Vt_all = np.linalg.svd(Xc_all, full_matrices=False)
proj_2d = Xc_all @ Vt_all[:2].T

X_tr_full, X_te_full, y_tr, y_te = train_test_split(
    X_sc,    y_enc, test_size=0.2, random_state=42, stratify=y_enc)
X_tr_pca,  X_te_pca,  _, _ = train_test_split(
    proj_2d,   y_enc, test_size=0.2, random_state=42, stratify=y_enc)
X_tr_umap, X_te_umap, _, _ = train_test_split(
    embedding, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

results = {}
for name, Xtr, Xte in [
        ('Full (20531 genes)', X_tr_full, X_te_full),
        ('2D PCA',             X_tr_pca,  X_te_pca),
        ('2D UMAP',            X_tr_umap, X_te_umap)]:
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(Xtr, y_tr)
    acc = clf.score(Xte, y_te)
    results[name] = acc
    print(f"  {name:25s}: test accuracy = {acc:.4f}")

print(f"{'Method':<22} {'Dimensions':<15} {'Test Accuracy'}")
print("-" * 48)
dims = {'Full (20531 genes)': 20531, '2D PCA': 2, '2D UMAP': 2}
for name, acc in results.items():
    print(f"{name:<22} {dims[name]:<15} {acc:.4f}")