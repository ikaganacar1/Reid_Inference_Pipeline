"""
Circular Buffer Implementation for ReID Gallery
Optimized for memory efficiency and fast access patterns
"""
import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class BufferEntry:
    """Single entry in circular buffer"""
    embedding: np.ndarray
    confidence: float
    timestamp: float
    frame_id: int


class CircularBuffer:
    """
    Circular buffer for storing ReID embeddings with automatic overwriting.
    
    Memory-efficient storage with O(1) insertion and access.
    Maintains last K observations for quality-aware aggregation.
    """
    
    def __init__(self, capacity: int = 10, embedding_dim: int = 2048):
        """
        Initialize circular buffer.
        
        Args:
            capacity: Maximum number of embeddings to store
            embedding_dim: Dimensionality of embeddings
        """
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        
        # Pre-allocate memory for efficiency
        self.embeddings = np.zeros((capacity, embedding_dim), dtype=np.float32)
        self.confidences = np.zeros(capacity, dtype=np.float32)
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.frame_ids = np.zeros(capacity, dtype=np.int32)
        
        self.size = 0
        self.write_idx = 0
        
    def add(self, embedding: np.ndarray, confidence: float, 
            timestamp: float, frame_id: int):
        """
        Add new embedding to buffer (overwrites oldest if full).
        
        Args:
            embedding: Feature vector (normalized)
            confidence: Detection confidence score
            timestamp: Time of observation
            frame_id: Frame number
        """
        self.embeddings[self.write_idx] = embedding
        self.confidences[self.write_idx] = confidence
        self.timestamps[self.write_idx] = timestamp
        self.frame_ids[self.write_idx] = frame_id
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    
    def get_centroid(self, 
                     quality_threshold: float = 0.0,
                     confidence_weighted: bool = True) -> Optional[np.ndarray]:
        """
        Compute centroid embedding from buffer.
        
        Args:
            quality_threshold: Minimum confidence to include in centroid
            confidence_weighted: Weight embeddings by confidence
            
        Returns:
            Aggregated centroid embedding (L2 normalized)
        """
        if self.size == 0:
            return None
        
        # Get valid embeddings
        valid_mask = self.confidences[:self.size] >= quality_threshold
        if not valid_mask.any():
            return None
        
        valid_embeddings = self.embeddings[:self.size][valid_mask]
        valid_confidences = self.confidences[:self.size][valid_mask]
        
        if confidence_weighted:
            # Weighted average
            weights = valid_confidences / valid_confidences.sum()
            centroid = np.average(valid_embeddings, axis=0, weights=weights)
        else:
            # Simple average
            centroid = np.mean(valid_embeddings, axis=0)
        
        # L2 normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
            
        return centroid
    
    def get_all_embeddings(self, quality_threshold: float = 0.0) -> np.ndarray:
        """Get all embeddings meeting quality threshold"""
        if self.size == 0:
            return np.array([])
        
        valid_mask = self.confidences[:self.size] >= quality_threshold
        return self.embeddings[:self.size][valid_mask]
    
    def get_latest(self, n: int = 1) -> List[BufferEntry]:
        """Get latest N entries"""
        if self.size == 0:
            return []
        
        n = min(n, self.size)
        entries = []
        
        for i in range(n):
            idx = (self.write_idx - 1 - i) % self.capacity
            if idx < self.size:
                entries.append(BufferEntry(
                    embedding=self.embeddings[idx],
                    confidence=self.confidences[idx],
                    timestamp=self.timestamps[idx],
                    frame_id=self.frame_ids[idx]
                ))
        
        return entries
    
    def get_statistics(self) -> dict:
        """Get buffer statistics"""
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }
        
        valid_confidences = self.confidences[:self.size]
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity,
            'avg_confidence': float(np.mean(valid_confidences)),
            'min_confidence': float(np.min(valid_confidences)),
            'max_confidence': float(np.max(valid_confidences))
        }
    
    def clear(self):
        """Clear all entries"""
        self.size = 0
        self.write_idx = 0
        self.embeddings.fill(0)
        self.confidences.fill(0)
        self.timestamps.fill(0)
        self.frame_ids.fill(0)
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity"""
        return self.size >= self.capacity
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return self.size == 0
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"CircularBuffer(size={stats['size']}/{stats['capacity']}, "
                f"avg_conf={stats['avg_confidence']:.3f})")
