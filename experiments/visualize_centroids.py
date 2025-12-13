import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import sys

# --- CONFIGURATION ---
# Path to your generated centroids file 

CENTROID_FILE = os.path.join('output', 'clusters', 'sbj_0_centroids.npy')
OUTPUT_IMAGE = os.path.join('output', 'visualizations', 'sbj_0_centroids_plot.png')

# Create output folder if missing
os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

def main():
    # 1. Load the Data
    if not os.path.exists(CENTROID_FILE):
        print(f"File not found: {CENTROID_FILE}")
        print("Please check the path or run the clustering step first.")
        sys.exit(1)
        
    centroids = np.load(CENTROID_FILE)
    print(f"Loaded centroids with shape: {centroids.shape}")
    # Expecting (100, 768)

    # 2. Reduce Dimensions to 2D
    
    # Method A: PCA (Good for global spread)
    pca = PCA(n_components=2)
    centroids_pca = pca.fit_transform(centroids)
    
    # Method B: t-SNE (Good for local grouping structure)
    
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='pca', learning_rate='auto')
    centroids_tsne = tsne.fit_transform(centroids)

    # 3. Plotting
    plt.figure(figsize=(16, 7))

    # Plot 1: PCA
    plt.subplot(1, 2, 1)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='blue', alpha=0.7, edgecolors='k')
    plt.title('Centroid Spread (PCA)\nGlobal Variance')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.grid(True, alpha=0.3)
    
    # Annotate a few points to see IDs
    for i, (x, y) in enumerate(centroids_pca):
        if i % 10 == 0: # Label every 10th cluster to avoid clutter
            plt.text(x, y, str(i), fontsize=9, ha='right')

    # Plot 2: t-SNE
    plt.subplot(1, 2, 2)
    plt.scatter(centroids_tsne[:, 0], centroids_tsne[:, 1], c='red', alpha=0.7, edgecolors='k')
    plt.title('Centroid Spread (t-SNE)\nLocal Grouping')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.grid(True, alpha=0.3)

    # Annotate same points
    for i, (x, y) in enumerate(centroids_tsne):
        if i % 10 == 0:
            plt.text(x, y, str(i), fontsize=9, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Visualization saved to: {OUTPUT_IMAGE}")
    print("Open this image to see how your 100 clusters are distributed.")

if __name__ == "__main__":
    main()
