import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from calligraphy_ai.dataset import preprocess_image_to_array
from calligraphy_ai.paths import DATA_DIR, PREPROCESSED_DIR


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def cache_path(source_path):
    relative = source_path.relative_to(DATA_DIR)
    return PREPROCESSED_DIR / f"{relative}.npy"


def preprocess_one(source_text):
    source_path = Path(source_text)
    destination = cache_path(source_path)
    if destination.exists():
        try:
            cached = np.load(destination, mmap_mode="r", allow_pickle=False)
            if cached.shape == (128, 128) and cached.dtype == np.uint8:
                return "cached"
        except (OSError, ValueError):
            pass

    processed = preprocess_image_to_array(str(source_path), target_size=128)
    encoded = np.rint(processed * 255).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, encoded, allow_pickle=False)
    os.replace(temporary, destination)
    return "created"


def main():
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    complete_marker = PREPROCESSED_DIR / ".complete"
    complete_marker.unlink(missing_ok=True)
    sources = sorted(
        path
        for phase in ("train", "test")
        for path in (DATA_DIR / phase).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    workers = min(8, os.cpu_count() or 1)
    counts = {"created": 0, "cached": 0}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(preprocess_one, map(str, sources), chunksize=32)
        for result in tqdm(results, total=len(sources), desc="Preprocessing"):
            counts[result] += 1

    manifest = {
        "source_count": len(sources),
        "created": counts["created"],
        "reused": counts["cached"],
        "image_shape": [128, 128],
        "dtype": "uint8",
    }
    (PREPROCESSED_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    complete_marker.write_text(json.dumps(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
