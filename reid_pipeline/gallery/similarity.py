"""
Similarity Computation and Re-ranking for Person ReID
Implements cosine similarity, Euclidean distance, and k-reciprocal encoding
"""
import numpy as np
from typing import Tuple, Optional
import logging


class SimilarityComputer:
    """
    Advanced similarity computation for ReID with re-ranking support.
    
    Methods:
    - Cosine similarity (preferred for modern deep learning)
    - Euclidean distance (for 512-D normalized embeddings)
    - k-reciprocal encoding for improved accuracy
    """
    
    def __init__(self, 
                 metric: str = 'cosine',
                 k_reciprocal: int = 20,
                 lambda_value: float = 0.3,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize similarity computer.
        
        Args:
            metric: 'cosine' or 'euclidean'
            k_reciprocal: k value for k-reciprocal encoding (20-25)
            lambda_value: Weight for local query expansion
        """
        self.metric = metric
        self.k = k_reciprocal
        self.lambda_value = lambda_value
        self.logger = logger or logging.getLogger(__name__)
        
    def cosine_similarity(self, 
                         query: np.ndarray, 
                         gallery: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and gallery.
        
        Args:
            query: Query embeddings (N, D) or (D,)
            gallery: Gallery embeddings (M, D)
            
        Returns:
            Similarity scores (N, M) or (M,)
        """
        # Handle single query
        if query.ndim == 1:
            query = query[np.newaxis, :]
            single_query = True
        else:
            single_query = False
        
        # L2 normalize
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        gallery_norm = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)
        
        # Dot product
        similarity = np.dot(query_norm, gallery_norm.T)
        
        return similarity[0] if single_query else similarity
    
    def euclidean_distance(self, 
                          query: np.ndarray, 
                          gallery: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance between query and gallery.
        
        Args:
            query: Query embeddings (N, D) or (D,)
            gallery: Gallery embeddings (M, D)
            
        Returns:
            Distance scores (N, M) or (M,) - lower is more similar
        """
        # Handle single query
        if query.ndim == 1:
            query = query[np.newaxis, :]
            single_query = True
        else:
            single_query = False
        
        # Compute pairwise distances
        # Using broadcasting: (N, 1, D) - (1, M, D) -> (N, M, D)
        diff = query[:, np.newaxis, :] - gallery[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        
        return distances[0] if single_query else distances
    
    def compute_similarity(self, 
                          query: np.ndarray, 
                          gallery: np.ndarray,
                          to_similarity: bool = True) -> np.ndarray:
        """
        Compute similarity using configured metric.
        
        Args:
            query: Query embeddings
            gallery: Gallery embeddings
            to_similarity: Convert distances to similarities (for euclidean)
            
        Returns:
            Similarity scores (higher is more similar)
        """
        if self.metric == 'cosine':
            return self.cosine_similarity(query, gallery)
        elif self.metric == 'euclidean':
            distances = self.euclidean_distance(query, gallery)
            if to_similarity:
                # Convert to similarity: s = 1 / (1 + d)
                return 1.0 / (1.0 + distances)
            return distances
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def k_reciprocal_encoding(self,
                             query_embeddings: np.ndarray,
                             gallery_embeddings: np.ndarray,
                             verbose: bool = False) -> np.ndarray:
        """
        Apply k-reciprocal encoding for re-ranking.
        
        This improves retrieval accuracy by checking mutual k-nearest neighbors.
        Typical improvement: +2-5% mAP
        Computational cost: 50-200ms per query
        
        Args:
            query_embeddings: Query embeddings (N, D)
            gallery_embeddings: Gallery embeddings (M, D)
            verbose: Print progress
            
        Returns:
            Re-ranked similarity matrix (N, M)
        """
        if verbose:
            self.logger.info("Computing k-reciprocal encoding...")
        
        N = query_embeddings.shape[0]
        M = gallery_embeddings.shape[0]
        
        # Compute initial similarity
        original_dist = self.euclidean_distance(query_embeddings, gallery_embeddings)
        
        # Compute gallery-gallery similarity for reciprocal neighbors
        gallery_dist = self.euclidean_distance(gallery_embeddings, gallery_embeddings)
        
        # Find k-reciprocal nearest neighbors
        initial_rank = np.argsort(original_dist, axis=1)
        gallery_rank = np.argsort(gallery_dist, axis=1)
        
        # Compute Jaccard distance
        jaccard_dist = np.zeros_like(original_dist)
        
        for i in range(N):
            # Forward k-nearest neighbors
            forward_k_neighbors = initial_rank[i, :self.k]
            
            # Check reciprocal relationship
            k_reciprocal_index = []
            for candidate in forward_k_neighbors:
                # Backward k-nearest neighbors
                backward_k_neighbors = gallery_rank[candidate, :self.k]
                
                # Reciprocal if query is in candidate's k-nearest neighbors
                if i < M and i in backward_k_neighbors:
                    k_reciprocal_index.append(candidate)
            
            if len(k_reciprocal_index) == 0:
                continue
            
            # Expand with reciprocal neighbors' neighbors
            k_reciprocal_expansion = k_reciprocal_index.copy()
            for candidate in k_reciprocal_index:
                candidate_neighbors = gallery_rank[candidate, :int(np.ceil(self.k / 2))]
                for neighbor in candidate_neighbors:
                    if neighbor in k_reciprocal_index:
                        k_reciprocal_expansion.append(neighbor)
            
            k_reciprocal_expansion = list(set(k_reciprocal_expansion))
            
            # Compute Jaccard distance
            for j in range(M):
                if j in k_reciprocal_expansion:
                    # Local query expansion
                    min_dist = np.min([original_dist[i, j]] + 
                                     [original_dist[i, k] for k in k_reciprocal_expansion])
                    jaccard_dist[i, j] = min_dist
                else:
                    jaccard_dist[i, j] = original_dist[i, j]
        
        # Final distance: weighted combination
        final_dist = jaccard_dist * (1 - self.lambda_value) + original_dist * self.lambda_value
        
        # Convert to similarity
        similarity = 1.0 / (1.0 + final_dist)
        
        if verbose:
            self.logger.info("k-reciprocal encoding complete")
        
        return similarity
    
    def batch_compute_top_k(self,
                           query_embeddings: np.ndarray,
                           gallery_embeddings: np.ndarray,
                           k: int = 5,
                           use_reciprocal: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute top-k matches for batch of queries.
        
        Args:
            query_embeddings: Query embeddings (N, D)
            gallery_embeddings: Gallery embeddings (M, D)
            k: Number of top matches to return
            use_reciprocal: Apply k-reciprocal encoding
            
        Returns:
            (top_k_indices, top_k_scores) both of shape (N, k)
        """
        if use_reciprocal:
            similarity = self.k_reciprocal_encoding(
                query_embeddings, 
                gallery_embeddings,
                verbose=False
            )
        else:
            similarity = self.compute_similarity(query_embeddings, gallery_embeddings)
        
        # Get top-k
        k = min(k, similarity.shape[1])
        top_k_indices = np.argsort(similarity, axis=1)[:, -k:][:, ::-1]
        top_k_scores = np.take_along_axis(similarity, top_k_indices, axis=1)
        
        return top_k_indices, top_k_scores
    
    def compute_similarity_stats(self, 
                                similarity_matrix: np.ndarray) -> dict:
        """Compute statistics on similarity matrix"""
        return {
            'mean_similarity': float(np.mean(similarity_matrix)),
            'max_similarity': float(np.max(similarity_matrix)),
            'min_similarity': float(np.min(similarity_matrix)),
            'std_similarity': float(np.std(similarity_matrix)),
            'median_similarity': float(np.median(similarity_matrix))
        }


class SpatialTemporalReranker:
    """
    Spatial-temporal re-ranking for multi-camera scenarios.
    
    Combines appearance similarity with spatial-temporal constraints.
    """
    
    def __init__(self, 
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 max_time_gap: float = 300.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize spatial-temporal re-ranker.
        
        Args:
            alpha: Weight for appearance similarity
            beta: Weight for spatial-temporal similarity
            max_time_gap: Maximum time gap for valid transitions (seconds)
        """
        self.alpha = alpha
        self.beta = beta
        self.max_time_gap = max_time_gap
        self.logger = logger or logging.getLogger(__name__)
        
        # Camera transition statistics (can be learned from data)
        self.transition_times = {}  # (cam_i, cam_j) -> mean_time
        self.transition_stds = {}   # (cam_i, cam_j) -> std_time
    
    def compute_spatial_temporal_score(self,
                                      time_gap: float,
                                      camera_pair: Tuple[int, int]) -> float:
        """
        Compute spatial-temporal likelihood score.
        
        Uses Weibull distribution or Gaussian for modeling transition times.
        
        Args:
            time_gap: Time difference between observations
            camera_pair: (source_camera, target_camera)
            
        Returns:
            Spatial-temporal score in [0, 1]
        """
        if time_gap > self.max_time_gap:
            return 0.0
        
        # Simple exponential decay model
        # Can be replaced with learned Weibull distributions
        decay_rate = 1.0 / 60.0  # Decay over 60 seconds
        score = np.exp(-decay_rate * time_gap)
        
        return float(score)
    
    def rerank(self,
              appearance_scores: np.ndarray,
              time_gaps: np.ndarray,
              camera_pairs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Re-rank using combined appearance and spatial-temporal scores.
        
        Args:
            appearance_scores: Appearance similarity scores (N, M)
            time_gaps: Time gaps between observations (N, M)
            camera_pairs: Camera pair information (N, M, 2) [optional]
            
        Returns:
            Combined scores (N, M)
        """
        N, M = appearance_scores.shape
        st_scores = np.zeros((N, M))
        
        for i in range(N):
            for j in range(M):
                if camera_pairs is not None:
                    cam_pair = tuple(camera_pairs[i, j])
                else:
                    cam_pair = (0, 0)  # Default
                
                st_scores[i, j] = self.compute_spatial_temporal_score(
                    time_gaps[i, j],
                    cam_pair
                )
        
        # Combined score
        combined_scores = self.alpha * appearance_scores + self.beta * st_scores
        
        return combined_scores
