from pathlib import Path

import numpy as np
from PIL import Image

from face_pca.pipeline import discover_images, fit_pca, load_image_matrix, reconstruction_mse


def test_image_loading_and_shape(tmp_path: Path) -> None:
    folder = tmp_path / "face1"
    folder.mkdir()
    Image.fromarray(np.full((8, 8), 64, dtype=np.uint8)).save(folder / "1.png")
    Image.fromarray(np.full((8, 8), 192, dtype=np.uint8)).save(folder / "2.png")

    paths = discover_images(tmp_path)
    matrix = load_image_matrix(paths, width=4, height=4)

    assert matrix.shape == (2, 16)
    assert np.all((matrix >= 0.0) & (matrix <= 1.0))


def test_more_components_reduce_reconstruction_error() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(20, 12))

    _, _, reconstructed_2 = fit_pca(matrix, components=2)
    _, _, reconstructed_8 = fit_pca(matrix, components=8)

    assert reconstruction_mse(matrix, reconstructed_8) <= reconstruction_mse(matrix, reconstructed_2)
