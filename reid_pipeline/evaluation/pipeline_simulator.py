"""
Pipeline Simulator for Dataset Evaluation

Simulates the full video pipeline (YOLO detection → ReID → Gallery matching)
on static dataset images to test gallery behavior and collect statistics.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EvaluationPipelineSimulator:
    """
    Simulates full pipeline processing on static images.

    Uses synthetic bounding boxes (full image) since Market-1501 images are pre-cropped.
    Feeds images through ReID extractor and gallery manager to simulate real-time processing.
    """

    def __init__(self,
                 reid_extractor,
                 gallery_manager,
                 use_synthetic_bbox: bool = True):
        """
        Initialize pipeline simulator.

        Args:
            reid_extractor: BatchReIDExtractor instance
            gallery_manager: GalleryManager instance
            use_synthetic_bbox: Use full-image bbox (True for pre-cropped datasets)
        """
        self.reid_extractor = reid_extractor
        self.gallery_manager = gallery_manager
        self.use_synthetic_bbox = use_synthetic_bbox

        # Performance tracking
        self.timings = {
            'reid_times': [],
            'gallery_times': []
        }

    def process_image(self,
                     image: np.ndarray,
                     camera_id: int,
                     frame_id: int) -> Dict:
        """
        Process single image through ReID and gallery matching.

        Args:
            image: RGB image (H, W, 3)
            camera_id: Camera ID for same-camera exclusion
            frame_id: Frame number for temporal tracking

        Returns:
            Dictionary with:
                - embedding: Feature vector
                - gallery_decision: (person_id, MatchDecision, similarity)
                - reid_time: Inference time
                - gallery_time: Matching time
        """
        # Create synthetic bbox (full image)
        h, w = image.shape[:2]
        bbox = np.array([0, 0, w, h], dtype=np.float32)

        # Extract ReID feature
        start_reid = time.time()
        embeddings, valid_flags = self.reid_extractor.extract_features_from_frame(
            image, [bbox]
        )
        reid_time = time.time() - start_reid

        if not valid_flags[0]:
            logger.warning(f"Failed to extract feature for frame {frame_id}")
            return None

        embedding = embeddings[0]

        # Match to gallery
        start_gallery = time.time()
        query_embeddings = np.array([embedding])
        query_confidences = np.array([1.0])  # Pre-cropped images have high confidence
        query_camera_ids = np.array([camera_id])

        matches = self.gallery_manager.match_queries_to_gallery(
            query_embeddings,
            query_confidences,
            query_camera_ids=query_camera_ids
        )
        gallery_time = time.time() - start_gallery

        # Record timings
        self.timings['reid_times'].append(reid_time)
        self.timings['gallery_times'].append(gallery_time)

        return {
            'embedding': embedding,
            'gallery_decision': matches[0],  # (person_id, MatchDecision, similarity)
            'reid_time': reid_time,
            'gallery_time': gallery_time
        }

    def build_gallery_from_images(self,
                                  images: List[Dict],
                                  progress_callback: Optional[Callable] = None) -> Dict:
        """
        Build gallery by processing images sequentially.
        Simulates frame-by-frame arrival in video processing.

        Args:
            images: List of dicts with keys: image, person_id, camera_id, file_path
            progress_callback: Optional callback(current, total, message)

        Returns:
            Statistics dictionary
        """
        logger.info(f"Building gallery from {len(images)} images...")

        total_images = len(images)
        for idx, img_data in enumerate(images):
            if 'image' not in img_data:
                continue

            # Process image
            result = self.process_image(
                img_data['image'],
                img_data['camera_id'],
                frame_id=idx
            )

            if result is None:
                continue

            # Add to gallery based on decision
            person_id, decision, similarity = result['gallery_decision']

            # Add new identities or update existing ones
            if person_id == -1:  # NEW decision
                person_id = self.gallery_manager.next_person_id
                self.gallery_manager.add_to_gallery(
                    person_id=person_id,
                    embedding=result['embedding'],
                    confidence=1.0,
                    frame_id=idx,
                    camera_id=img_data['camera_id'],
                    force_add=True
                )
            else:  # MATCH or UNCERTAIN - update existing
                entry = self.gallery_manager.gallery.get(person_id)
                if entry:
                    entry.embedding_buffer.add(
                        result['embedding'], 1.0, time.time(), idx
                    )
                    entry.update_centroid()
                    entry.update_last_seen(idx)

            # Progress update
            if progress_callback and (idx % 100 == 0 or idx == total_images - 1):
                progress_callback(idx + 1, total_images, f"Building gallery: {idx+1}/{total_images}")

        stats = self.gallery_manager.get_statistics()
        logger.info(f"Gallery built: {stats['gallery_size']} identities")

        return stats

    def evaluate_queries(self,
                        query_images: List[Dict],
                        progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Process query images and record gallery decisions.

        Args:
            query_images: List of dicts with: image, person_id, camera_id, filename
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of result dictionaries per query
        """
        logger.info(f"Evaluating {len(query_images)} queries...")

        results = []
        total_queries = len(query_images)

        for idx, query_data in enumerate(query_images):
            if 'image' not in query_data:
                continue

            # Process query
            result = self.process_image(
                query_data['image'],
                query_data['camera_id'],
                frame_id=idx + 10000  # Offset to avoid conflict with gallery frames
            )

            if result is None:
                continue

            person_id, decision, similarity = result['gallery_decision']

            # Record result
            results.append({
                'query_id': query_data.get('filename', f'query_{idx}'),
                'query_person_id': query_data['person_id'],
                'query_camera_id': query_data['camera_id'],
                'embedding': result['embedding'],
                'matched_person_id': person_id,
                'decision': decision,
                'similarity': similarity,
                'reid_time': result['reid_time'],
                'gallery_time': result['gallery_time']
            })

            # Progress update
            if progress_callback and (idx % 50 == 0 or idx == total_queries - 1):
                progress_callback(idx + 1, total_queries, f"Processing queries: {idx+1}/{total_queries}")

        logger.info(f"Evaluated {len(results)} queries")
        return results

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        reid_times = np.array(self.timings['reid_times'])
        gallery_times = np.array(self.timings['gallery_times'])

        total_time = reid_times.sum() + gallery_times.sum()
        fps = len(reid_times) / total_time if total_time > 0 else 0

        return {
            'avg_reid_time': float(reid_times.mean()) if len(reid_times) > 0 else 0,
            'avg_gallery_time': float(gallery_times.mean()) if len(gallery_times) > 0 else 0,
            'total_time': float(total_time),
            'fps': float(fps),
            'total_processed': len(reid_times)
        }
