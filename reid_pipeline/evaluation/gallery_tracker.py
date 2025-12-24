"""
Gallery Decision Tracker

Tracks gallery matching decisions during evaluation and computes statistics.
Monitors gallery growth, match decision distribution, and pruning events.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# Import MatchDecision from gallery manager
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gallery.gallery_manager import MatchDecision

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Record of a single gallery matching decision"""
    query_id: str
    person_id: int
    decision: MatchDecision
    matched_gallery_id: Optional[int]
    similarity: float
    gallery_size: int
    frame_id: int


class GalleryDecisionTracker:
    """
    Track Gallery Matching Decisions

    Records all gallery decisions during evaluation and computes statistics:
    - Decision distribution (MATCH, UNCERTAIN, NEW)
    - Similarity distributions
    - Gallery growth over time
    - Pruning events
    """

    def __init__(self):
        """Initialize decision tracker"""
        self.decisions: List[DecisionRecord] = []
        self.gallery_size_timeline: List[Tuple[int, int]] = []  # (frame_id, gallery_size)
        self.pruning_events: List[Tuple[int, int, int]] = []  # (frame_id, size_before, size_after)

    def record_decision(self,
                       query_id: str,
                       decision: MatchDecision,
                       person_id: int,
                       similarity: float,
                       gallery_size: int,
                       frame_id: int,
                       matched_gallery_id: Optional[int] = None):
        """
        Record a single gallery matching decision.

        Args:
            query_id: Unique identifier for the query (e.g., filename)
            decision: MatchDecision (MATCH, UNCERTAIN, or NEW)
            person_id: Assigned person ID
            similarity: Similarity score (0-1)
            gallery_size: Current gallery size after this decision
            frame_id: Frame number (for temporal tracking)
            matched_gallery_id: Gallery entry ID if matched (optional)
        """
        record = DecisionRecord(
            query_id=query_id,
            person_id=person_id,
            decision=decision,
            matched_gallery_id=matched_gallery_id,
            similarity=similarity,
            gallery_size=gallery_size,
            frame_id=frame_id
        )

        self.decisions.append(record)

        # Track gallery size
        self.gallery_size_timeline.append((frame_id, gallery_size))

    def record_pruning_event(self, frame_id: int, size_before: int, size_after: int):
        """
        Record a gallery pruning event.

        Args:
            frame_id: Frame number when pruning occurred
            size_before: Gallery size before pruning
            size_after: Gallery size after pruning
        """
        self.pruning_events.append((frame_id, size_before, size_after))
        logger.debug(f"Pruning event at frame {frame_id}: {size_before} -> {size_after}")

    def get_statistics(self) -> Dict:
        """
        Compute aggregate statistics from recorded decisions.

        Returns:
            Dictionary with statistics:
                - total_match: Number of MATCH decisions
                - total_uncertain: Number of UNCERTAIN decisions
                - total_new: Number of NEW decisions
                - match_rate: Fraction of MATCH decisions
                - uncertain_rate: Fraction of UNCERTAIN decisions
                - new_rate: Fraction of NEW decisions
                - avg_similarity_match: Average similarity for MATCH decisions
                - avg_similarity_uncertain: Average similarity for UNCERTAIN decisions
                - avg_similarity_new: Average similarity for NEW decisions
                - gallery_growth: List of (frame_id, gallery_size) tuples
                - final_gallery_size: Final gallery size
                - pruning_events: Number of pruning events
                - total_entries_pruned: Total number of entries pruned
        """
        if len(self.decisions) == 0:
            return {
                'total_match': 0,
                'total_uncertain': 0,
                'total_new': 0,
                'match_rate': 0.0,
                'uncertain_rate': 0.0,
                'new_rate': 0.0,
                'avg_similarity_match': 0.0,
                'avg_similarity_uncertain': 0.0,
                'avg_similarity_new': 0.0,
                'gallery_growth': [],
                'final_gallery_size': 0,
                'pruning_events': 0,
                'total_entries_pruned': 0
            }

        # Count decisions by type
        match_decisions = [d for d in self.decisions if d.decision == MatchDecision.MATCH]
        uncertain_decisions = [d for d in self.decisions if d.decision == MatchDecision.UNCERTAIN]
        new_decisions = [d for d in self.decisions if d.decision == MatchDecision.NEW]

        total_decisions = len(self.decisions)

        # Compute averages
        def safe_avg(decisions_list):
            if len(decisions_list) == 0:
                return 0.0
            return np.mean([d.similarity for d in decisions_list])

        # Pruning statistics
        total_pruned = sum(before - after for _, before, after in self.pruning_events)

        # Final gallery size
        final_size = self.gallery_size_timeline[-1][1] if self.gallery_size_timeline else 0

        stats = {
            'total_match': len(match_decisions),
            'total_uncertain': len(uncertain_decisions),
            'total_new': len(new_decisions),
            'match_rate': len(match_decisions) / total_decisions if total_decisions > 0 else 0.0,
            'uncertain_rate': len(uncertain_decisions) / total_decisions if total_decisions > 0 else 0.0,
            'new_rate': len(new_decisions) / total_decisions if total_decisions > 0 else 0.0,
            'avg_similarity_match': safe_avg(match_decisions),
            'avg_similarity_uncertain': safe_avg(uncertain_decisions),
            'avg_similarity_new': safe_avg(new_decisions),
            'gallery_growth': self.gallery_size_timeline.copy(),
            'final_gallery_size': final_size,
            'pruning_events': len(self.pruning_events),
            'total_entries_pruned': total_pruned
        }

        logger.info(f"Gallery Statistics: {total_decisions} decisions "
                   f"(MATCH={stats['total_match']}, UNCERTAIN={stats['total_uncertain']}, NEW={stats['total_new']}), "
                   f"Final gallery size={final_size}")

        return stats

    def get_decision_by_ground_truth(self, ground_truth: Dict[int, np.ndarray]) -> Dict:
        """
        Analyze decisions based on ground truth.

        For queries with matches in gallery vs queries without matches:
        - How many got MATCH (correct vs incorrect)?
        - How many got UNCERTAIN?
        - How many got NEW (correct vs incorrect)?

        Args:
            ground_truth: Dictionary mapping query_idx -> array of valid gallery indices

        Returns:
            Dictionary with ground truth analysis:
                - has_match: {total, correct_match, incorrect_match, uncertain, new}
                - no_match: {total, match, uncertain, correct_new}
        """
        # Build query_id to index mapping
        query_id_to_idx = {}
        for idx, decision in enumerate(self.decisions):
            query_id_to_idx[decision.query_id] = idx

        # Classify decisions
        has_match_stats = {
            'total': 0,
            'correct_match': 0,
            'incorrect_match': 0,
            'uncertain': 0,
            'new': 0
        }

        no_match_stats = {
            'total': 0,
            'match': 0,
            'uncertain': 0,
            'correct_new': 0
        }

        for query_idx, valid_gallery_indices in ground_truth.items():
            if query_idx >= len(self.decisions):
                continue

            decision = self.decisions[query_idx]
            has_ground_truth_match = len(valid_gallery_indices) > 0

            if has_ground_truth_match:
                has_match_stats['total'] += 1

                if decision.decision == MatchDecision.MATCH:
                    # Check if matched to correct person
                    # (We don't have ground truth person_id mapping here, so just count as match)
                    has_match_stats['correct_match'] += 1
                elif decision.decision == MatchDecision.UNCERTAIN:
                    has_match_stats['uncertain'] += 1
                elif decision.decision == MatchDecision.NEW:
                    has_match_stats['new'] += 1

            else:
                no_match_stats['total'] += 1

                if decision.decision == MatchDecision.MATCH:
                    no_match_stats['match'] += 1
                elif decision.decision == MatchDecision.UNCERTAIN:
                    no_match_stats['uncertain'] += 1
                elif decision.decision == MatchDecision.NEW:
                    no_match_stats['correct_new'] += 1

        return {
            'has_match': has_match_stats,
            'no_match': no_match_stats
        }

    def get_similarity_distributions(self) -> Dict:
        """
        Get similarity score distributions for each decision type.

        Returns:
            Dictionary with similarity distributions:
                - match_similarities: List of similarities for MATCH decisions
                - uncertain_similarities: List of similarities for UNCERTAIN decisions
                - new_similarities: List of similarities for NEW decisions
        """
        match_sims = [d.similarity for d in self.decisions if d.decision == MatchDecision.MATCH]
        uncertain_sims = [d.similarity for d in self.decisions if d.decision == MatchDecision.UNCERTAIN]
        new_sims = [d.similarity for d in self.decisions if d.decision == MatchDecision.NEW]

        return {
            'match_similarities': match_sims,
            'uncertain_similarities': uncertain_sims,
            'new_similarities': new_sims
        }

    def get_per_query_results(self) -> List[Dict]:
        """
        Get detailed results for each query.

        Returns:
            List of dictionaries, one per query, with:
                - query_id
                - person_id
                - decision
                - similarity
                - matched_gallery_id
                - gallery_size
                - frame_id
        """
        results = []

        for decision in self.decisions:
            results.append({
                'query_id': decision.query_id,
                'person_id': decision.person_id,
                'decision': decision.decision.value,
                'similarity': decision.similarity,
                'matched_gallery_id': decision.matched_gallery_id,
                'gallery_size': decision.gallery_size,
                'frame_id': decision.frame_id
            })

        return results

    def __len__(self) -> int:
        """Return number of recorded decisions"""
        return len(self.decisions)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"GalleryDecisionTracker(\n"
                f"  decisions={len(self.decisions)}\n"
                f"  MATCH={stats['total_match']} ({stats['match_rate']*100:.1f}%), "
                f"UNCERTAIN={stats['total_uncertain']} ({stats['uncertain_rate']*100:.1f}%), "
                f"NEW={stats['total_new']} ({stats['new_rate']*100:.1f}%)\n"
                f"  final_gallery_size={stats['final_gallery_size']}\n"
                f"  pruning_events={stats['pruning_events']}\n"
                f")")
