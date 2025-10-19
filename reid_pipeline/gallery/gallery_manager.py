"""
Enhanced Gallery Manager with Dynamic Pruning and Three-Tier Matching
Production-grade implementation for NVIDIA Jetson deployment
"""
import numpy as np
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
import logging
from enum import Enum

from .circular_buffer import CircularBuffer


class MatchDecision(Enum):
    """Three-tier matching classification"""
    MATCH = "match"           # High similarity - associate with existing ID
    UNCERTAIN = "uncertain"   # Medium similarity - needs temporal analysis
    NEW = "new"              # Low similarity - create new gallery entry


@dataclass
class GalleryEntry:
    """
    Gallery entry with circular buffer for dynamic representation.
    
    Memory footprint per entry (with 2048-D embeddings, 10-frame buffer):
    - Circular buffer: 2048 * 10 * 4 bytes = 81,920 bytes (~80KB)
    - Metadata: ~1KB
    Total: ~81KB per identity
    
    For 100 identities: ~8.1MB
    """
    person_id: int
    embedding_buffer: CircularBuffer
    centroid_embedding: np.ndarray
    confidence: float
    first_seen: float
    last_seen: float
    frames_since_seen: int = 0
    appearance_count: int = 0
    track_state: str = "active"  # active, lost, deleted
    
    # Temporal tracking
    last_bbox: Optional[np.ndarray] = None
    last_frame_id: int = -1
    
    # Quality metrics
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    
    def update_last_seen(self, frame_id: int):
        """Update last seen information"""
        self.last_seen = time.time()
        self.last_frame_id = frame_id
        self.frames_since_seen = 0
        self.appearance_count += 1
    
    def increment_absent_frames(self):
        """Increment frames since last seen"""
        self.frames_since_seen += 1
    
    def update_centroid(self, quality_threshold: float = 0.7):
        """Recompute centroid from buffer"""
        new_centroid = self.embedding_buffer.get_centroid(
            quality_threshold=quality_threshold,
            confidence_weighted=True
        )
        if new_centroid is not None:
            self.centroid_embedding = new_centroid
            
            # Update quality metrics
            stats = self.embedding_buffer.get_statistics()
            self.avg_confidence = stats['avg_confidence']
            self.max_confidence = stats['max_confidence']


class GalleryManager:
    """
    Dynamic Gallery Management with Adaptive Pruning.
    
    Features:
    - Circular buffer storage for each identity
    - Three-tier similarity matching
    - Time-based and capacity-based pruning
    - EMA smoothing with confidence weighting
    - Quality-aware gallery admission
    """
    
    def __init__(self,
                 max_gallery_size: int = 500,
                 buffer_capacity: int = 10,
                 embedding_dim: int = 2048,
                 similarity_threshold_match: float = 0.70,
                 similarity_threshold_new: float = 0.50,
                 quality_admission_threshold: float = 0.91,
                 ttl_frames_crowded: int = 30,
                 ttl_frames_sparse: int = 150,
                 ema_alpha_high_conf: float = 0.3,
                 ema_alpha_low_conf: float = 0.1,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Gallery Manager.
        
        Args:
            max_gallery_size: Maximum number of identities in gallery
            buffer_capacity: Size of circular buffer per identity (5-10 frames)
            embedding_dim: Dimensionality of embeddings (512/2048)
            similarity_threshold_match: Threshold for MATCH decision (0.70-0.75)
            similarity_threshold_new: Threshold for NEW decision (<0.50)
            quality_admission_threshold: Min confidence for gallery admission (0.91-0.94)
            ttl_frames_crowded: Time-to-live in crowded scenarios (30 frames)
            ttl_frames_sparse: Time-to-live in sparse scenarios (150-300 frames)
            ema_alpha_high_conf: EMA alpha for high confidence updates
            ema_alpha_low_conf: EMA alpha for low confidence updates
        """
        self.max_gallery_size = max_gallery_size
        self.buffer_capacity = buffer_capacity
        self.embedding_dim = embedding_dim
        
        # Three-tier thresholds
        self.threshold_match = similarity_threshold_match
        self.threshold_new = similarity_threshold_new
        
        # Quality control
        self.quality_threshold = quality_admission_threshold
        
        # Pruning parameters
        self.ttl_crowded = ttl_frames_crowded
        self.ttl_sparse = ttl_frames_sparse
        self.current_ttl = ttl_frames_sparse  # Start with sparse assumption
        
        # EMA parameters
        self.ema_alpha_high = ema_alpha_high_conf
        self.ema_alpha_low = ema_alpha_low_conf
        
        # Gallery storage
        self.gallery: Dict[int, GalleryEntry] = {}
        self.next_person_id = 0
        self.current_frame = 0
        
        # Statistics
        self.stats = {
            'total_added': 0,
            'total_pruned': 0,
            'match_decisions': 0,
            'uncertain_decisions': 0,
            'new_decisions': 0,
            'gallery_size_history': []
        }
        
        # Logging
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"GalleryManager initialized: max_size={max_gallery_size}, "
                        f"buffer_capacity={buffer_capacity}, "
                        f"thresholds=[{similarity_threshold_new:.2f}, "
                        f"{similarity_threshold_match:.2f}]")
    
    def compute_similarity_batch(self, 
                                 query_embeddings: np.ndarray,
                                 gallery_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between queries and gallery (batch).
        
        Args:
            query_embeddings: Shape (N, D) - query embeddings
            gallery_embeddings: Shape (M, D) - gallery embeddings
            
        Returns:
            Similarity matrix of shape (N, M)
        """
        # Ensure L2 normalization
        query_norm = query_embeddings / (np.linalg.norm(
            query_embeddings, axis=1, keepdims=True) + 1e-8)
        gallery_norm = gallery_embeddings / (np.linalg.norm(
            gallery_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity via dot product
        similarity_matrix = np.dot(query_norm, gallery_norm.T)
        
        return similarity_matrix
    
    def match_queries_to_gallery(self,
                                 query_embeddings: np.ndarray,
                                 query_confidences: np.ndarray) -> List[Tuple[int, MatchDecision, float]]:
        """
        Match query embeddings to gallery using three-tier classification.
        
        Args:
            query_embeddings: Query embeddings (N, D)
            query_confidences: Detection confidences (N,)
            
        Returns:
            List of (person_id or -1, decision, similarity_score) for each query
        """
        if len(self.gallery) == 0:
            # Empty gallery - all queries are NEW
            return [(-1, MatchDecision.NEW, 0.0) for _ in range(len(query_embeddings))]
        
        # Get gallery centroids
        gallery_ids = list(self.gallery.keys())
        gallery_centroids = np.array([
            self.gallery[pid].centroid_embedding 
            for pid in gallery_ids
        ])
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_batch(
            query_embeddings, gallery_centroids
        )
        
        # Match each query
        results = []
        for i in range(len(query_embeddings)):
            similarities = similarity_matrix[i]
            max_similarity = np.max(similarities)
            best_match_idx = np.argmax(similarities)
            best_person_id = gallery_ids[best_match_idx]
            
            # Three-tier decision
            if max_similarity >= self.threshold_match:
                decision = MatchDecision.MATCH
                person_id = best_person_id
                self.stats['match_decisions'] += 1
            elif max_similarity >= self.threshold_new:
                decision = MatchDecision.UNCERTAIN
                person_id = best_person_id  # Tentative match
                self.stats['uncertain_decisions'] += 1
            else:
                decision = MatchDecision.NEW
                person_id = -1
                self.stats['new_decisions'] += 1
            
            results.append((person_id, decision, float(max_similarity)))
        
        return results
    
    def add_to_gallery(self,
                      person_id: int,
                      embedding: np.ndarray,
                      confidence: float,
                      frame_id: int,
                      bbox: Optional[np.ndarray] = None,
                      force_add: bool = False) -> bool:
        """
        Add new identity to gallery with quality control.
        
        Args:
            person_id: Identity ID
            embedding: Feature vector
            confidence: Detection confidence
            frame_id: Current frame number
            bbox: Bounding box
            force_add: Skip quality check
            
        Returns:
            True if added successfully
        """
        # Quality check
        if not force_add and confidence < self.quality_threshold:
            self.logger.debug(f"Rejected gallery addition: confidence {confidence:.3f} "
                            f"< threshold {self.quality_threshold:.3f}")
            return False
        
        # Check capacity
        if len(self.gallery) >= self.max_gallery_size:
            self.logger.warning(f"Gallery at capacity ({self.max_gallery_size}), "
                              "pruning required before adding new entry")
            self.prune_gallery(force_prune_count=1)
        
        # Create new entry
        buffer = CircularBuffer(
            capacity=self.buffer_capacity,
            embedding_dim=self.embedding_dim
        )
        buffer.add(embedding, confidence, time.time(), frame_id)
        
        entry = GalleryEntry(
            person_id=person_id,
            embedding_buffer=buffer,
            centroid_embedding=embedding.copy(),
            confidence=confidence,
            first_seen=time.time(),
            last_seen=time.time(),
            last_bbox=bbox,
            last_frame_id=frame_id,
            avg_confidence=confidence,
            max_confidence=confidence
        )
        
        self.gallery[person_id] = entry
        self.stats['total_added'] += 1
        
        if person_id >= self.next_person_id:
            self.next_person_id = person_id + 1
        
        self.logger.info(f"Added person {person_id} to gallery "
                        f"(confidence={confidence:.3f}, gallery_size={len(self.gallery)})")
        
        return True
    
    def update_gallery_entry(self,
                            person_id: int,
                            embedding: np.ndarray,
                            confidence: float,
                            frame_id: int,
                            bbox: Optional[np.ndarray] = None,
                            use_ema: bool = True):
        """
        Update existing gallery entry with EMA smoothing.
        
        Args:
            person_id: Identity to update
            embedding: New feature vector
            confidence: Detection confidence
            frame_id: Current frame number
            bbox: Bounding box
            use_ema: Apply EMA smoothing
        """
        if person_id not in self.gallery:
            self.logger.warning(f"Attempted to update non-existent person {person_id}")
            return
        
        entry = self.gallery[person_id]
        
        # Add to circular buffer
        entry.embedding_buffer.add(embedding, confidence, time.time(), frame_id)
        
        # Update centroid with EMA or recompute
        if use_ema:
            # Confidence-weighted EMA
            alpha = self.ema_alpha_high if confidence > 0.7 else self.ema_alpha_low
            entry.centroid_embedding = (alpha * embedding + 
                                       (1 - alpha) * entry.centroid_embedding)
            
            # Renormalize
            norm = np.linalg.norm(entry.centroid_embedding)
            if norm > 0:
                entry.centroid_embedding /= norm
        else:
            # Recompute from buffer
            entry.update_centroid(quality_threshold=0.7)
        
        # Update metadata
        entry.update_last_seen(frame_id)
        entry.confidence = max(entry.confidence, confidence)
        entry.last_bbox = bbox
        
        self.logger.debug(f"Updated person {person_id} "
                         f"(appearances={entry.appearance_count}, "
                         f"confidence={confidence:.3f})")
    
    def prune_gallery(self, force_prune_count: Optional[int] = None):
        """
        Prune gallery using adaptive TTL and LRU eviction.
        
        Two strategies:
        1. Time-based: Remove entries not seen for > TTL frames
        2. Capacity-based: LRU eviction when at max capacity
        
        Args:
            force_prune_count: Force prune this many entries (for capacity management)
        """
        if len(self.gallery) == 0:
            return
        
        # Adaptive TTL based on gallery density
        gallery_utilization = len(self.gallery) / self.max_gallery_size
        if gallery_utilization > 0.8:
            self.current_ttl = self.ttl_crowded
        else:
            self.current_ttl = self.ttl_sparse
        
        # Time-based pruning
        to_remove = []
        for person_id, entry in self.gallery.items():
            entry.increment_absent_frames()
            
            if entry.frames_since_seen > self.current_ttl:
                to_remove.append(person_id)
                entry.track_state = "deleted"
        
        # Capacity-based pruning (LRU)
        if force_prune_count is not None and force_prune_count > 0:
            # Sort by last seen time (oldest first)
            sorted_entries = sorted(
                self.gallery.items(),
                key=lambda x: x[1].last_seen
            )
            
            for person_id, entry in sorted_entries[:force_prune_count]:
                if person_id not in to_remove:
                    to_remove.append(person_id)
                    entry.track_state = "deleted"
        
        # Remove entries
        for person_id in to_remove:
            del self.gallery[person_id]
            self.stats['total_pruned'] += 1
        
        if to_remove:
            self.logger.info(f"Pruned {len(to_remove)} entries from gallery "
                           f"(TTL={self.current_ttl}, remaining={len(self.gallery)})")
    
    def update_frame(self):
        """Update frame counter and trigger pruning if needed"""
        self.current_frame += 1
        
        # Periodic pruning (every 10 frames)
        if self.current_frame % 10 == 0:
            self.prune_gallery()
        
        # Track gallery size history
        if self.current_frame % 30 == 0:
            self.stats['gallery_size_history'].append(len(self.gallery))
    
    def get_statistics(self) -> dict:
        """Get comprehensive gallery statistics"""
        if len(self.gallery) == 0:
            return {
                'gallery_size': 0,
                'max_size': self.max_gallery_size,
                'utilization': 0.0,
                'total_added': self.stats['total_added'],
                'total_pruned': self.stats['total_pruned'],
                'match_decisions': self.stats['match_decisions'],
                'uncertain_decisions': self.stats['uncertain_decisions'],
                'new_decisions': self.stats['new_decisions'],
                'current_ttl': self.current_ttl,
                'avg_confidence': 0.0,
                'avg_appearances': 0.0
            }
        
        confidences = [e.avg_confidence for e in self.gallery.values()]
        appearances = [e.appearance_count for e in self.gallery.values()]
        
        return {
            'gallery_size': len(self.gallery),
            'max_size': self.max_gallery_size,
            'utilization': len(self.gallery) / self.max_gallery_size,
            'total_added': self.stats['total_added'],
            'total_pruned': self.stats['total_pruned'],
            'match_decisions': self.stats['match_decisions'],
            'uncertain_decisions': self.stats['uncertain_decisions'],
            'new_decisions': self.stats['new_decisions'],
            'current_ttl': self.current_ttl,
            'avg_confidence': float(np.mean(confidences)),
            'max_confidence': float(np.max(confidences)),
            'min_confidence': float(np.min(confidences)),
            'avg_appearances': float(np.mean(appearances)),
            'max_appearances': int(np.max(appearances))
        }
    
    def save_gallery(self, filepath: Path):
        """Save gallery state to disk"""
        gallery_data = {
            'gallery': {},
            'next_person_id': self.next_person_id,
            'current_frame': self.current_frame,
            'stats': self.stats,
            'config': {
                'max_gallery_size': self.max_gallery_size,
                'buffer_capacity': self.buffer_capacity,
                'embedding_dim': self.embedding_dim,
                'threshold_match': self.threshold_match,
                'threshold_new': self.threshold_new
            }
        }
        
        # Serialize gallery entries
        for person_id, entry in self.gallery.items():
            gallery_data['gallery'][person_id] = {
                'person_id': entry.person_id,
                'centroid_embedding': entry.centroid_embedding,
                'confidence': entry.confidence,
                'first_seen': entry.first_seen,
                'last_seen': entry.last_seen,
                'appearance_count': entry.appearance_count,
                # Note: Not saving full circular buffer to save space
                # Only centroid is persisted
            }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(gallery_data, f)
        
        self.logger.info(f"Gallery saved to {filepath} ({len(self.gallery)} entries)")
    
    def load_gallery(self, filepath: Path) -> bool:
        """Load gallery state from disk"""
        if not filepath.exists():
            self.logger.warning(f"Gallery file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                gallery_data = pickle.load(f)
            
            # Restore gallery entries (recreate buffers)
            for person_id, data in gallery_data['gallery'].items():
                buffer = CircularBuffer(
                    capacity=self.buffer_capacity,
                    embedding_dim=self.embedding_dim
                )
                
                # Add centroid to buffer as initial entry
                buffer.add(
                    data['centroid_embedding'],
                    data['confidence'],
                    data['last_seen'],
                    0
                )
                
                entry = GalleryEntry(
                    person_id=data['person_id'],
                    embedding_buffer=buffer,
                    centroid_embedding=data['centroid_embedding'],
                    confidence=data['confidence'],
                    first_seen=data['first_seen'],
                    last_seen=data['last_seen'],
                    appearance_count=data['appearance_count']
                )
                
                self.gallery[person_id] = entry
            
            self.next_person_id = gallery_data['next_person_id']
            self.current_frame = gallery_data.get('current_frame', 0)
            self.stats.update(gallery_data.get('stats', {}))
            
            self.logger.info(f"Gallery loaded from {filepath} ({len(self.gallery)} entries)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading gallery: {e}")
            return False
    
    def clear(self):
        """Clear entire gallery"""
        self.gallery.clear()
        self.next_person_id = 0
        self.current_frame = 0
        self.stats = {
            'total_added': 0,
            'total_pruned': 0,
            'match_decisions': 0,
            'uncertain_decisions': 0,
            'new_decisions': 0,
            'gallery_size_history': []
        }
        self.logger.info("Gallery cleared")
