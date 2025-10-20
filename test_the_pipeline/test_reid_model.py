#!/usr/bin/env python3
"""
Test Script to Verify ReID Model is Working Correctly

This script tests if your ReID model produces discriminative embeddings.
If the model is not working, this will show you immediately.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_reid_model(model_path):
    """Test if ReID model produces good embeddings"""
    
    print("=" * 80)
    print("ReID Model Quality Test")
    print("=" * 80)
    print()
    
    # Import after path is set
    from reid_pipeline.models.reid_model import BatchReIDExtractor
    import cv2
    
    print(f"Testing model: {model_path}")
    print()
    
    # Check if model file exists
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ ERROR: Model file not found: {model_path}")
        print(f"   Please check the path and try again.")
        return False
    
    print(f"✓ Model file exists: {model_file.name} ({model_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print()
    
    # Load extractor
    print("Loading ReID extractor...")
    try:
        extractor = BatchReIDExtractor(
            model_path=str(model_path),
            embedding_dim=2048,
            batch_size=8,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✓ Extractor loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading extractor: {e}")
        return False
    
    print()
    print("-" * 80)
    print("Test 1: Embedding Extraction")
    print("-" * 80)
    
    # Create test image
    test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Create identical crops (simulating same person)
    bbox_same1 = np.array([100, 100, 300, 500])
    bbox_same2 = np.array([100, 100, 300, 500])
    
    # Create different crops (simulating different people)
    bbox_diff1 = np.array([400, 150, 600, 550])
    bbox_diff2 = np.array([700, 200, 900, 600])
    bbox_diff3 = np.array([1000, 250, 1200, 650])
    
    bboxes = [bbox_same1, bbox_same2, bbox_diff1, bbox_diff2, bbox_diff3]
    
    print(f"Extracting embeddings from {len(bboxes)} crops...")
    embeddings, valid = extractor.extract_features_from_frame(test_img, bboxes)
    
    if embeddings.shape[0] != len(bboxes):
        print(f"❌ ERROR: Expected {len(bboxes)} embeddings, got {embeddings.shape[0]}")
        return False
    
    print(f"✓ Extracted {embeddings.shape[0]} embeddings of shape {embeddings.shape}")
    print()
    
    print("-" * 80)
    print("Test 2: Embedding Discrimination")
    print("-" * 80)
    
    # Compute cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embeddings)
    
    print("Similarity Matrix:")
    print("     ", "  ".join([f"P{i}" for i in range(len(bboxes))]))
    for i in range(len(bboxes)):
        row_str = f"P{i}: " + "  ".join([f"{sims[i,j]:.3f}" for j in range(len(bboxes))])
        print(row_str)
    print()
    
    # Analyze results
    print("Analysis:")
    print(f"  Same person similarity (P0 vs P1): {sims[0,1]:.4f}")
    print(f"  Different person similarities:")
    print(f"    P0 vs P2: {sims[0,2]:.4f}")
    print(f"    P0 vs P3: {sims[0,3]:.4f}")
    print(f"    P0 vs P4: {sims[0,4]:.4f}")
    print()
    
    # Check if embeddings are discriminative
    same_person_sim = sims[0, 1]
    diff_person_sims = [sims[0, 2], sims[0, 3], sims[0, 4]]
    avg_diff_sim = np.mean(diff_person_sims)
    
    print("Expected behavior:")
    print("  - Same person similarity should be ~1.0 (identical crops)")
    print("  - Different person similarities should be <0.7")
    print()
    
    issues = []
    
    if same_person_sim < 0.95:
        issues.append(f"Same person similarity is too low ({same_person_sim:.4f})")
    
    if avg_diff_sim > 0.7:
        issues.append(f"Different person similarity is too high ({avg_diff_sim:.4f})")
    
    # Check if all embeddings are too similar (random model problem)
    if avg_diff_sim > 0.6 and same_person_sim < 0.8:
        issues.append("All embeddings are too similar - model may be using random weights!")
    
    print("-" * 80)
    print("Test Results")
    print("-" * 80)
    
    if len(issues) == 0:
        print("✓ Model appears to be working correctly!")
        print("  Embeddings are discriminative and should work for ReID.")
        return True
    else:
        print("❌ PROBLEMS DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("This suggests your ReID model may not be properly trained or loaded.")
        print("Common causes:")
        print("  1. Model file is corrupted")
        print("  2. Model architecture mismatch")
        print("  3. Model was not trained properly")
        print("  4. Wrong embedding_dim specified")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ReID model quality')
    parser.add_argument('--model', '-m', 
                       default='test_the_pipeline/reid_ltcc.pth',
                       help='Path to ReID model')
    
    args = parser.parse_args()
    
    success = test_reid_model(args.model)
    
    print()
    print("=" * 80)
    if success:
        print("✓ TEST PASSED - Model is working correctly")
    else:
        print("❌ TEST FAILED - Model needs attention")
    print("=" * 80)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
