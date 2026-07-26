"""
ReID Dataset Loader
Supports Market1501 and LTCC format datasets
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image


class ReIDDataset:
    """
    Dataset loader for Market1501-format person ReID datasets.

    Supports:
    - Market1501: {pid}_{camid}_s{seqid}_{frame}.jpg (e.g., 0001_c1s1_001051_00.jpg)
    - LTCC: {pid}_{camid}s{seqid}_{frame}_{cloth}.jpg (e.g., 0001_c11s1_33_00.jpg)

    Attributes:
        images: List of image paths
        pids: Person IDs (numpy array)
        camids: Camera IDs (numpy array)
    """

    # Regex patterns for different formats
    PATTERNS = {
        'market1501': re.compile(r'(\d+)_c(\d+)s(\d+)_(\d+)_(\d+)\.(?:jpg|png|jpeg)', re.IGNORECASE),
        'ltcc': re.compile(r'(\d+)_c(\d+)s(\d+)_(\d+)_(\d+)\.(?:jpg|png|jpeg)', re.IGNORECASE),
    }

    def __init__(self,
                 root: Path,
                 split: str = 'query',
                 dataset_format: str = 'ltcc',
                 extensions: List[str] = None):
        """
        Initialize dataset loader.

        Args:
            root: Root directory containing dataset splits
            split: Dataset split ('query', 'bounding_box_test', 'bounding_box_train')
            dataset_format: Dataset format ('market1501' or 'ltcc')
            extensions: Valid image extensions
        """
        self.root = Path(root)
        self.split = split
        self.dataset_format = dataset_format
        self.extensions = extensions or ['.jpg', '.png', '.jpeg']

        self.split_dir = self.root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Load dataset
        self.images: List[Path] = []
        self.pids: np.ndarray = np.array([])
        self.camids: np.ndarray = np.array([])
        self.clothes: np.ndarray = np.array([])  # For LTCC

        self._load_dataset()

    def _load_dataset(self):
        """Load and parse all images in the split directory."""
        images = []
        pids = []
        camids = []
        clothes = []

        # Get all image files
        all_files = sorted(self.split_dir.iterdir())

        for img_path in all_files:
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in self.extensions:
                continue

            # Parse filename
            parsed = self._parse_filename(img_path.name)
            if parsed is None:
                continue

            pid, camid, cloth = parsed

            # Skip junk images (pid = -1 or 0 in some datasets)
            if pid < 0:
                continue

            images.append(img_path)
            pids.append(pid)
            camids.append(camid)
            clothes.append(cloth)

        self.images = images
        self.pids = np.array(pids, dtype=np.int32)
        self.camids = np.array(camids, dtype=np.int32)
        self.clothes = np.array(clothes, dtype=np.int32)

        print(f"Loaded {len(self.images)} images from {self.split}")
        print(f"  Unique PIDs: {len(np.unique(self.pids))}")
        print(f"  Unique Cameras: {len(np.unique(self.camids))}")

    def _parse_filename(self, filename: str) -> Optional[Tuple[int, int, int]]:
        """
        Parse filename to extract person ID, camera ID, and clothing ID.

        Args:
            filename: Image filename

        Returns:
            Tuple of (pid, camid, cloth_id) or None if parsing fails
        """
        # Try LTCC/Market format: {pid}_{camid}s{seq}_{frame}_{cloth}.jpg
        pattern = self.PATTERNS.get(self.dataset_format, self.PATTERNS['ltcc'])
        match = pattern.match(filename)

        if match:
            pid = int(match.group(1))
            camid = int(match.group(2))
            cloth = int(match.group(5))  # Last group is clothing/frame suffix
            return pid, camid, cloth

        # Fallback: try simple pattern
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                pid = int(parts[0])
                # Extract camera from c{N} format
                cam_match = re.search(r'c(\d+)', parts[1])
                camid = int(cam_match.group(1)) if cam_match else 0
                cloth = 0
                return pid, camid, cloth
            except (ValueError, AttributeError):
                pass

        return None

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, int]:
        """
        Get image and metadata by index.

        Args:
            idx: Image index

        Returns:
            Tuple of (image, pid, camid)
        """
        img_path = self.images[idx]
        img = self._read_bgr(img_path)

        return img, self.pids[idx], self.camids[idx]

    def get_image(self, idx: int) -> np.ndarray:
        """Load single image by index."""
        return self._read_bgr(self.images[idx])

    @staticmethod
    def _read_bgr(path: Path) -> np.ndarray:
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"Failed to load image: {path}") from exc
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def get_batch_images(self, indices: List[int]) -> List[np.ndarray]:
        """Load batch of images by indices."""
        return [self.get_image(idx) for idx in indices]

    def get_unique_pids(self) -> np.ndarray:
        """Get array of unique person IDs."""
        return np.unique(self.pids)

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'split': self.split,
            'num_images': len(self.images),
            'num_pids': len(np.unique(self.pids)),
            'num_cameras': len(np.unique(self.camids)),
            'images_per_pid': len(self.images) / max(len(np.unique(self.pids)), 1),
        }


def load_query_gallery(root: Path,
                       dataset_format: str = 'ltcc',
                       query_dir: str = 'query',
                       gallery_dir: str = 'bounding_box_test') -> Tuple[ReIDDataset, ReIDDataset]:
    """
    Load query and gallery datasets.

    Args:
        root: Dataset root directory
        dataset_format: Format ('market1501' or 'ltcc')
        query_dir: Query directory name
        gallery_dir: Gallery directory name

    Returns:
        Tuple of (query_dataset, gallery_dataset)
    """
    query = ReIDDataset(root, split=query_dir, dataset_format=dataset_format)
    gallery = ReIDDataset(root, split=gallery_dir, dataset_format=dataset_format)

    return query, gallery


if __name__ == "__main__":
    # Test dataset loading
    import sys

    root = Path("data")
    if not root.exists():
        print(f"ERROR: Data directory not found: {root}")
        sys.exit(1)

    print("Loading LTCC dataset...")
    print("=" * 60)

    # Load query
    print("\n[Query Set]")
    query = ReIDDataset(root, split='query', dataset_format='ltcc')
    print(f"Stats: {query.get_stats()}")

    # Load gallery
    print("\n[Gallery Set]")
    gallery = ReIDDataset(root, split='bounding_box_test', dataset_format='ltcc')
    print(f"Stats: {gallery.get_stats()}")

    # Test loading an image
    print("\n[Sample Image Test]")
    img, pid, camid = query[0]
    print(f"Image shape: {img.shape}")
    print(f"PID: {pid}, CamID: {camid}")
    print(f"Image path: {query.images[0]}")

    print("\nDataset loading test passed!")
