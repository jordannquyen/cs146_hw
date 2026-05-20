import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
PALETTE = sns.color_palette('tab10')

url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
print(f"Shape: {penguins.shape}")
print("\nDescribe:\n", penguins.describe())
print("\nSpecies counts:\n", penguins['species'].value_counts())
print("\nIsland counts:\n", penguins['island'].value_counts())

print("\nMissing values per column:\n", penguins.isnull().sum())
num_cols = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']
penguins_clean = penguins.dropna(subset=num_cols)
print(f"\nPenguins after dropping NaN rows: {len(penguins_clean)}")

fig, ax = plt.subplots(figsize=(7, 5))
species_list = penguins_clean['species'].unique()
colors_sp = {'Adelie': PALETTE[0], 'Chinstrap': PALETTE[1], 'Gentoo': PALETTE[2]}
for sp in species_list:
    sub = penguins_clean[penguins_clean['species'] == sp]
    ax.scatter(sub['flipper_length_mm'], sub['bill_length_mm'],
               c=[colors_sp[sp]], label=sp, alpha=0.7, s=40)
ax.set_xlabel('Flipper Length (mm)')
ax.set_ylabel('Bill Length (mm)')
ax.set_title('Penguin Morphology by Species')
ax.legend()
plt.tight_layout()
plt.savefig('homework3/figures/p3a_scatter_species.png')
plt.close()
X_raw = penguins_clean[num_cols].values
print(f"shape: {X_raw.shape}")

mu = X_raw.mean(axis=0)
sigma = X_raw.std(axis=0)
X_std = (X_raw - mu) / sigma
print(f"\nFeature means:  {dict(zip(num_cols, mu.round(3)))}")
print(f"Feature stds:   {dict(zip(num_cols, sigma.round(3)))}")
print(f"Standardized means (should be ~0): {X_std.mean(axis=0).round(6)}")
print(f"Standardized stds  (should be ~1): {X_std.std(axis=0).round(6)}")

def kmeans(X, k, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    n= X.shape[0]
    init_idx  = rng.choice(n, k, replace=False)
    centroids = X[init_idx].copy()
    labels    = np.zeros(n, dtype=int)

    for iteration in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if iteration > 0 and np.all(new_labels == labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = X[mask].mean(axis=0)
    dists_final = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    inertia= float(np.sum(np.min(dists_final, axis=1) ** 2))
    return centroids, labels, inertia


def _kmeans_j_history(X, k, max_iter=100, seed=0):
    rng       = np.random.default_rng(seed)
    n         = X.shape[0]
    centroids = X[rng.choice(n, k, replace=False)].copy()
    labels    = np.zeros(n, dtype=int)
    J_history = []

    for iteration in range(max_iter):
        dists      = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        J_history.append(float(np.sum(np.min(dists, axis=1) ** 2)))

        if iteration > 0 and np.all(new_labels == labels):
            labels = new_labels
            break
        labels = new_labels

        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = X[mask].mean(axis=0)

    return J_history

centroids, labels, inertia = kmeans(X_std, k=3, seed=0)
print("3(c)(i) — kmeans(X_std, k=3, seed=0)")

print(f"\nFinal inertia J = {inertia:.4f}")

print(f"\nFinal labels  (shape {labels.shape}):")
print(f"  {labels}")

print(f"\nCluster sizes:")
for c in range(3):
    print(f"  Cluster {c}: {(labels == c).sum()} points")

print(f"\nFinal centroids  (shape {centroids.shape}, standardized):")
header = f"  {'Cluster':>8}  {'bill_len':>10}  {'bill_dep':>10}  {'flipper':>10}  {'body_mass':>10}"
print(header)
for c in range(3):
    vals = "  ".join(f"{v:+.4f}" for v in centroids[c])
    print(f"  {c:>8}  {vals}")

J_hist = _kmeans_j_history(X_std, k=3, seed=0)
print(f"\n  {'Iter':>5}  {'J':>12}  {'Change':>12}  {'J(t)≤J(t-1)?':>14}")
print("  " + "-" * 48)
for i, j in enumerate(J_hist):
    if i == 0:
        print(f"  {i:>5}  {j:>12.4f}  {'—':>12}  {'—':>14}")
    else:
        delta = j - J_hist[i - 1]
        ok    = "✓" if delta <= 0 else "VIOLATION"
        print(f"  {i:>5}  {j:>12.4f}  {delta:>+12.4f}  {ok:>14}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(len(J_hist)), J_hist,
        marker='o', color='steelblue', linewidth=2, markersize=7)
for i, j in enumerate(J_hist):
    ax.annotate(f'{j:.1f}', (i, j),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=8)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Inertia J', fontsize=12)
ax.set_title('K-Means Objective J vs. Iteration  (k=3, seed=0)', fontsize=13)
ax.set_xticks(range(len(J_hist)))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('homework3/figures/p3c_inertia_iterations.png', dpi=130)
plt.close()

centroids3, labels3, inertia3 = kmeans(X_std, k=3, seed=0)
print(f"Final inertia (k=3): {inertia3:.4f}")

species_true = penguins_clean['species'].values
ct = pd.crosstab(labels3, species_true, rownames=['Cluster'], colnames=['Species'])
print("\nCross-tabulation:\n", ct)

total_correct = 0
for c in range(3):
    mask = labels3 == c
    if mask.sum() > 0:
        counts = pd.Series(species_true[mask]).value_counts()
        total_correct += counts.iloc[0]
purity = total_correct / len(labels3)
print(f"\nClustering Purity: {purity:.4f} ({purity*100:.1f}%)")

centroids_orig = centroids3 * sigma + mu
profile_df = pd.DataFrame(centroids_orig, columns=num_cols)
print("\nCluster centroids in original units:")
print(profile_df.round(2))
print("\nCluster profiles:")
for i, row in profile_df.iterrows():
    print(f"  Cluster {i}: bill={row['bill_length_mm']:.1f}mm len / "
          f"{row['bill_depth_mm']:.1f}mm depth, "
          f"flipper={row['flipper_length_mm']:.1f}mm, "
          f"body={row['body_mass_g']:.0f}g")
fig, ax = plt.subplots(figsize=(7, 5))
colors_cl = {0: PALETTE[0], 1: PALETTE[1], 2: PALETTE[2]}
for c in range(3):
    mask = labels3 == c
    sub = penguins_clean.iloc[mask]
    ax.scatter(sub['flipper_length_mm'], sub['bill_length_mm'],
               c=[colors_cl[c]], label=f'Cluster {c}', alpha=0.7, s=40)
ax.set_xlabel('Flipper Length (mm)')
ax.set_ylabel('Bill Length (mm)')
ax.set_title('K-Means Clusters (k=3) – Penguins')
ax.legend()
plt.tight_layout()
plt.savefig('homework3/figures/p3d_scatter_clusters.png')
plt.close()

print("e")
k_range = range(1, 11)
mean_inertias = []
for k in k_range:
    seeds_inertia = [kmeans(X_std, k, seed=s)[2] for s in range(10)]  # [2] =inertia
    mean_inertias.append(np.mean(seeds_inertia))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(k_range), mean_inertias, marker='o', color=PALETTE[0])
ax.set_xlabel('k')
ax.set_ylabel('Mean Inertia (10 seeds)')
ax.set_title('Elbow Method – K-Means Inertia vs. k')
ax.axvline(3, color='red', linestyle='--', label='k=3 (elbow)')
ax.legend()
plt.tight_layout()
plt.savefig('homework3/figures/p3e_elbow.png')
plt.close()
print(f"Mean inertias: {[round(j,2) for j in mean_inertias]}")

print("\n── (f) silhouete")
sil_scores = []
for k in range(2, 11):
    best_inertia = np.inf
    best_labels  = None
    for s in range(10):
        _, lab, inert = kmeans(X_std, k, seed=s)
        if inert < best_inertia:
            best_inertia = inert
            best_labels  = lab
    sc = silhouette_score(X_std, best_labels)
    sil_scores.append(sc)
    print(f"  k={k}: silhouette={sc:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(2, 11), sil_scores, marker='o', color=PALETTE[1])
ax.set_xlabel('k')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score vs. k')
best_k_sil = np.argmax(sil_scores) + 2
ax.axvline(best_k_sil, color='red', linestyle='--', label=f'k={best_k_sil}')
ax.legend()
plt.tight_layout()
plt.savefig('homework3/figures/p3f_silhouette.png')
plt.close()

print("g")
inertias_50 = []
for s in range(50):
    _, _, inert = kmeans(X_std, k=3, seed=s)
    inertias_50.append(inert)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(inertias_50, bins=15, color=PALETTE[2], edgecolor='white')
ax.set_xlabel('Final Inertia')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Final Inertia over 50 Seeds (k=3)')
plt.tight_layout()
plt.savefig('homework3/figures/p3g_inertia_hist.png')
plt.close()
print(f"Inertia range: min={min(inertias_50):.2f}, max={max(inertias_50):.2f}, "
      f"std={np.std(inertias_50):.2f}")

best_seed  = np.argmin(inertias_50)
worst_seed = np.argmax(inertias_50)
_, lab_best,  _ = kmeans(X_std, k=3, seed=best_seed)
_, lab_worst, _ = kmeans(X_std, k=3, seed=worst_seed)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, lab, title in zip(axes, [lab_best, lab_worst],
                           [f'Best seed {best_seed} (J={min(inertias_50):.2f})',
                            f'Worst seed {worst_seed} (J={max(inertias_50):.2f})']):
    for c in range(3):
        mask = lab == c
        sub  = penguins_clean.iloc[mask]
        ax.scatter(sub['flipper_length_mm'], sub['bill_length_mm'],
                   c=[PALETTE[c]], label=f'Cluster {c}', alpha=0.7, s=30)
    ax.set_xlabel('Flipper Length (mm)')
    ax.set_ylabel('Bill Length (mm)')
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.savefig('homework3/figures/p3g_best_worst.png')
plt.close()