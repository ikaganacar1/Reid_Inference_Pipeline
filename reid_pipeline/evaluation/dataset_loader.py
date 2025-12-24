"""
Market-1501 Dataset Loader

Loads Market-1501 dataset with proper query/gallery splits and ground truth generation.
Handles filename parsing and same-camera match exclusion.

Market-1501 Format:
- Filename: PPPP_cC_sSSSS_FFFFFF_DD.jpg
  - PPPP: Person ID (0001-1501, 0000=distractor, -1=junk)
  - C: Camera ID (1-6)
  - SSSS: Sequence ID
  - FFFFFF: Frame number
  - DD: Detection ID (multiple crops per frame)

Dataset Structure:
- query/: 3,368 query images (750 identities)
- bounding_box_test/: 19,732 gallery images (750 identities + distractors)
- bounding_box_train/: 12,936 training images (751 identities)
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata extracted from Market-1501 filename"""
    file_path: Path
    filename: str
    person_id: int
    camera_id: int
    sequence_id: int
    frame_id: int
    detection_id: int
    is_distractor: bool  # person_id == 0
    is_junk: bool  # person_id == -1


class Market1501Dataset:
    """
    Market-1501 Dataset Loader

    Loads query and gallery sets with proper splits and metadata.
    Builds ground truth mapping for evaluation.
    """

    def __init__(self, dataset_path: Path, load_images: bool = False):
        """
        Initialize Market-1501 dataset loader.

        Args:
            dataset_path: Root path to Market-1501 dataset
            load_images: If True, load images into memory (use with caution)
        """
        self.dataset_path = Path(dataset_path)
        self.load_images = load_images

        # Dataset directories
        self.query_dir = self.dataset_path / "query"
        self.gallery_dir = self.dataset_path / "bounding_box_test"
        self.train_dir = self.dataset_path / "bounding_box_train"

        # Validate dataset structure
        self._validate_dataset()

        # Load dataset
        self.query_metadata: List[ImageMetadata] = []
        self.gallery_metadata: List[ImageMetadata] = []
        self.train_metadata: List[ImageMetadata] = []

        logger.info(f"Loading Market-1501 dataset from: {self.dataset_path}")
        self._load_dataset()

        # Build ground truth
        self.ground_truth = self._build_ground_truth()

        logger.info(f"Dataset loaded: {len(self.query_metadata)} queries, "
                   f"{len(self.gallery_metadata)} gallery, "
                   f"{len(self.train_metadata)} train")

    def _validate_dataset(self):
        """Validate dataset directory structure"""
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

        if not self.query_dir.exists():
            raise ValueError(f"Query directory not found: {self.query_dir}")

        if not self.gallery_dir.exists():
            raise ValueError(f"Gallery directory not found: {self.gallery_dir}")

        # Train directory is optional
        if not self.train_dir.exists():
            logger.warning(f"Train directory not found: {self.train_dir}")

    @staticmethod
    def parse_filename(filename: str) -> Optional[ImageMetadata]:
        """
        Parse Market-1501 filename format.

        Format: PPPP_cC_sSSSS_FFFFFF_DD.jpg

        Args:
            filename: Image filename

        Returns:
            ImageMetadata object or None if parsing fails

        Examples:
            >>> Market1501Dataset.parse_filename("0001_c2_s3_000451_03.jpg")
            ImageMetadata(person_id=1, camera_id=2, sequence_id=3, frame_id=451, detection_id=3)
        """
        # Pattern: PPPP_cC_sSSSS_FFFFFF_DD.jpg
        pattern = r'(-?\d+)_c(\d+)s(\d+)_(\d+)_(\d+)'
        match = re.match(pattern, filename)

        if not match:
            logger.warning(f"Failed to parse filename: {filename}")
            return None

        person_id = int(match.group(1))
        camera_id = int(match.group(2))
        sequence_id = int(match.group(3))
        frame_id = int(match.group(4))
        detection_id = int(match.group(5))

        return ImageMetadata(
            file_path=Path(filename),
            filename=filename,
            person_id=person_id,
            camera_id=camera_id,
            sequence_id=sequence_id,
            frame_id=frame_id,
            detection_id=detection_id,
            is_distractor=(person_id == 0),
            is_junk=(person_id == -1)
        )

    def _load_images_from_dir(self, directory: Path) -> List[ImageMetadata]:
        """
        Load all images from a directory.

        Args:
            directory: Directory containing images

        Returns:
            List of ImageMetadata objects
        """
        metadata_list = []

        if not directory.exists():
            return metadata_list

        # Get all .jpg files
        image_files = sorted(directory.glob("*.jpg"))

        for img_path in image_files:
            metadata = self.parse_filename(img_path.name)
            if metadata is not None:
                # Update file_path to absolute path
                metadata.file_path = img_path
                metadata_list.append(metadata)

        return metadata_list

    def _load_dataset(self):
        """Load all dataset splits"""
        # Load query set
        self.query_metadata = self._load_images_from_dir(self.query_dir)
        logger.info(f"Loaded {len(self.query_metadata)} query images")

        # Load gallery set
        self.gallery_metadata = self._load_images_from_dir(self.gallery_dir)

        # Filter out junk images from gallery
        self.gallery_metadata = [m for m in self.gallery_metadata if not m.is_junk]
        logger.info(f"Loaded {len(self.gallery_metadata)} gallery images (junk excluded)")

        # Load train set (optional)
        self.train_metadata = self._load_images_from_dir(self.train_dir)
        logger.info(f"Loaded {len(self.train_metadata)} train images")

    def _build_ground_truth(self) -> Dict[int, np.ndarray]:
        """
        Build ground truth mapping for evaluation.

        For each query, identify valid gallery matches:
        - Same person_id
        - Different camera_id (exclude same-camera matches)
        - Not distractor (person_id != 0)

        Returns:
            Dictionary mapping query_idx -> array of valid gallery indices
        """
        ground_truth = {}

        for query_idx, query_meta in enumerate(self.query_metadata):
            valid_gallery_indices = []

            for gallery_idx, gallery_meta in enumerate(self.gallery_metadata):
                # Check if this is a valid match
                is_same_person = (gallery_meta.person_id == query_meta.person_id)
                is_different_camera = (gallery_meta.camera_id != query_meta.camera_id)
                is_not_distractor = (not gallery_meta.is_distractor)

                if is_same_person and is_different_camera and is_not_distractor:
                    valid_gallery_indices.append(gallery_idx)

            ground_truth[query_idx] = np.array(valid_gallery_indices, dtype=np.int32)

        # Log statistics
        num_queries_with_matches = sum(1 for indices in ground_truth.values() if len(indices) > 0)
        avg_matches_per_query = np.mean([len(indices) for indices in ground_truth.values()])

        logger.info(f"Ground truth built: {num_queries_with_matches}/{len(self.query_metadata)} "
                   f"queries have matches (avg {avg_matches_per_query:.1f} matches/query)")

        return ground_truth

    def load_image(self, metadata: ImageMetadata) -> np.ndarray:
        """
        Load image from disk.

        Args:
            metadata: Image metadata

        Returns:
            Image as numpy array (H, W, 3) in RGB format
        """
        img = Image.open(metadata.file_path).convert('RGB')
        return np.array(img)

    def get_queries(self, with_images: bool = False) -> List[Dict]:
        """
        Get query set with metadata.

        Args:
            with_images: If True, load images into memory

        Returns:
            List of dictionaries with keys:
                - metadata: ImageMetadata
                - image: np.ndarray (if with_images=True)
                - person_id: int
                - camera_id: int
        """
        queries = []

        for meta in self.query_metadata:
            query_dict = {
                'metadata': meta,
                'person_id': meta.person_id,
                'camera_id': meta.camera_id,
                'file_path': str(meta.file_path),
                'filename': meta.filename
            }

            if with_images:
                query_dict['image'] = self.load_image(meta)

            queries.append(query_dict)

        return queries

    def get_gallery(self, with_images: bool = False) -> List[Dict]:
        """
        Get gallery set with metadata.

        Args:
            with_images: If True, load images into memory

        Returns:
            List of dictionaries with keys:
                - metadata: ImageMetadata
                - image: np.ndarray (if with_images=True)
                - person_id: int
                - camera_id: int
        """
        gallery = []

        for meta in self.gallery_metadata:
            gallery_dict = {
                'metadata': meta,
                'person_id': meta.person_id,
                'camera_id': meta.camera_id,
                'file_path': str(meta.file_path),
                'filename': meta.filename,
                'is_distractor': meta.is_distractor
            }

            if with_images:
                gallery_dict['image'] = self.load_image(meta)

            gallery.append(gallery_dict)

        return gallery

    def get_train(self, with_images: bool = False) -> List[Dict]:
        """
        Get training set with metadata.

        Args:
            with_images: If True, load images into memory

        Returns:
            List of dictionaries with metadata
        """
        train = []

        for meta in self.train_metadata:
            train_dict = {
                'metadata': meta,
                'person_id': meta.person_id,
                'camera_id': meta.camera_id,
                'file_path': str(meta.file_path),
                'filename': meta.filename
            }

            if with_images:
                train_dict['image'] = self.load_image(meta)

            train.append(train_dict)

        return train

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        query_person_ids = set(m.person_id for m in self.query_metadata)
        gallery_person_ids = set(m.person_id for m in self.gallery_metadata if not m.is_distractor)

        num_queries_with_matches = sum(1 for indices in self.ground_truth.values() if len(indices) > 0)

        return {
            'num_queries': len(self.query_metadata),
            'num_gallery': len(self.gallery_metadata),
            'num_train': len(self.train_metadata),
            'num_query_identities': len(query_person_ids),
            'num_gallery_identities': len(gallery_person_ids),
            'num_cameras': 6,  # Market-1501 has 6 cameras
            'num_queries_with_matches': num_queries_with_matches,
            'avg_matches_per_query': np.mean([len(indices) for indices in self.ground_truth.values()])
        }

    def __len__(self) -> int:
        """Return total number of images (query + gallery + train)"""
        return len(self.query_metadata) + len(self.gallery_metadata) + len(self.train_metadata)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"Market1501Dataset(\n"
                f"  queries={stats['num_queries']}, "
                f"  gallery={stats['num_gallery']}, "
                f"  train={stats['num_train']}\n"
                f"  identities: query={stats['num_query_identities']}, "
                f"gallery={stats['num_gallery_identities']}\n"
                f"  avg_matches_per_query={stats['avg_matches_per_query']:.1f}\n"
                f")")
