from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def discover_images(root: Path) -> list[Path]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no supported images found under {root}")
    return paths


def load_image_matrix(paths: list[Path], width: int = 128, height: int = 128) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    rows: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L").resize((width, height)), dtype=np.float32)
        rows.append(array.reshape(-1) / 255.0)
    return np.vstack(rows)


def fit_pca(matrix: np.ndarray, components: int) -> tuple[PCA, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    max_components = min(matrix.shape)
    if not 1 <= components <= max_components:
        raise ValueError(f"components must be between 1 and {max_components}")

    pca = PCA(n_components=components, svd_solver="full")
    embeddings = pca.fit_transform(matrix)
    reconstructed = pca.inverse_transform(embeddings)
    return pca, embeddings, reconstructed


def reconstruction_mse(matrix: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(mean_squared_error(matrix, reconstructed))


def save_variance_curve(pca: PCA, output: Path) -> None:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o")
    axis.set_xlabel("Principal components")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_eigenfaces(pca: PCA, width: int, height: int, output: Path, limit: int = 8) -> None:
    count = min(limit, len(pca.components_))
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(pca.components_[index].reshape(height, width), cmap="gray")
        axis.set_title(f"PC {index + 1}")

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_reconstructions(
    matrix: np.ndarray,
    reconstructed: np.ndarray,
    width: int,
    height: int,
    output: Path,
    limit: int = 5,
) -> None:
    count = min(limit, len(matrix))
    figure, axes = plt.subplots(count, 2, figsize=(6, 2.5 * count), squeeze=False)

    for index in range(count):
        axes[index, 0].imshow(matrix[index].reshape(height, width), cmap="gray", vmin=0, vmax=1)
        axes[index, 0].set_title("Original")
        axes[index, 1].imshow(
            reconstructed[index].reshape(height, width),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[index, 1].set_title("PCA reconstruction")
        axes[index, 0].axis("off")
        axes[index, 1].axis("off")

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run(root: Path, output: Path, components: int, width: int, height: int) -> None:
    paths = discover_images(root)
    matrix = load_image_matrix(paths, width=width, height=height)
    pca, embeddings, reconstructed = fit_pca(matrix, components=components)

    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "component": np.arange(1, components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(output / "explained_variance.csv", index=False)

    embedding_frame = pd.DataFrame(
        embeddings,
        columns=[f"pc_{index + 1}" for index in range(components)],
    )
    embedding_frame.insert(0, "image", [str(path.relative_to(root)) for path in paths])
    embedding_frame.to_csv(output / "embeddings.csv", index=False)

    distances = squareform(pdist(embeddings, metric="euclidean"))
    pd.DataFrame(distances, index=embedding_frame["image"], columns=embedding_frame["image"]).to_csv(
        output / "distance_matrix.csv"
    )

    save_variance_curve(pca, output / "variance_curve.png")
    save_eigenfaces(pca, width, height, output / "eigenfaces.png")
    save_reconstructions(matrix, reconstructed, width, height, output / "reconstruction_examples.png")

    print(f"images: {len(paths)}")
    print(f"pixel dimensions: {matrix.shape[1]}")
    print(f"components: {components}")
    print(f"cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"reconstruction MSE: {reconstruction_mse(matrix, reconstructed):.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCA/eigenface image representation analysis")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--components", type=int, default=20)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.input, args.output, args.components, args.width, args.height)


if __name__ == "__main__":
    main()
