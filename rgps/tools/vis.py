import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

matplotlib.use("Agg")


def plot_selection_pca(features, property, property_name, indices, output_path):
    """Generates a 2D-PCA plot colored by given property."""
    if features.shape[0] < 2:
        print(f"PCA dimensionality reduction failed: No enough dimensions.")
        return

    try:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(features)
        bv = np.array(property, dtype=float)
        valid_mask = np.isfinite(bv)
        vmin, vmax = None, None

        if np.any(valid_mask):
            try:
                vmin, vmax = np.quantile(bv[valid_mask], [0.01, 0.99])
            except Exception:
                vmin, vmax = np.nanmin(bv[valid_mask]), np.nanmax(bv[valid_mask])

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=bv,
            cmap="viridis",
            s=15,
            alpha=0.7,
            vmin=vmin,
            vmax=vmax,
            label="Candidates",
            rasterized=True,
        )

        if indices:
            sel_coords = coords[indices]
            ax.scatter(
                sel_coords[:, 0],
                sel_coords[:, 1],
                s=45,
                facecolors="none",
                edgecolors="r",
                linewidths=1.5,
                label="Selected",
            )

        plt.colorbar(sc, label=f"{property_name}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("Descriptor Space")
        ax.legend()

        plt.tight_layout()
        fig_path = Path(output_path) / "disc_mapping.png"
        fig.savefig(fig_path, dpi=600)
        plt.close(fig)

    except Exception as e:
        print(f"PCA plotting failed: {e}.")
