"""
Unit Test for Market-1501 Dataset Loader

Tests filename parsing, dataset loading, and ground truth generation
with synthetic test data.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid_pipeline.evaluation.dataset_loader import Market1501Dataset, ImageMetadata


def create_synthetic_dataset(dataset_path: Path, num_queries: int = 10, num_gallery: int = 50):
    """
    Create a synthetic Market-1501 dataset for testing.

    Args:
        dataset_path: Root directory for synthetic dataset
        num_queries: Number of query images to create
        num_gallery: Number of gallery images to create
    """
    # Create directories
    query_dir = dataset_path / "query"
    gallery_dir = dataset_path / "bounding_box_test"
    train_dir = dataset_path / "bounding_box_train"

    query_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic query images
    # Use person IDs 1-5, cameras 1-6
    for i in range(num_queries):
        person_id = (i % 5) + 1  # Person IDs: 1, 2, 3, 4, 5
        camera_id = (i % 6) + 1  # Camera IDs: 1-6
        sequence_id = 1
        frame_id = i * 100
        detection_id = 1

        filename = f"{person_id:04d}_c{camera_id}s{sequence_id}_{frame_id:06d}_{detection_id:02d}.jpg"
        img_path = query_dir / filename

        # Create a random RGB image
        img = Image.fromarray(np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8))
        img.save(img_path)

    print(f"Created {num_queries} query images")

    # Create synthetic gallery images
    # Include matches for queries (different cameras) + distractors
    for i in range(num_gallery):
        if i < num_queries * 3:  # Create 3 gallery images per query (different cameras)
            query_idx = i // 3
            person_id = (query_idx % 5) + 1
            camera_id = ((i % 5) + 2)  # Different camera from query
            if camera_id > 6:
                camera_id = camera_id - 6
        else:  # Create distractors
            person_id = 0  # Distractor
            camera_id = (i % 6) + 1

        sequence_id = 1
        frame_id = i * 100
        detection_id = 1

        filename = f"{person_id:04d}_c{camera_id}s{sequence_id}_{frame_id:06d}_{detection_id:02d}.jpg"
        img_path = gallery_dir / filename

        # Create a random RGB image
        img = Image.fromarray(np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8))
        img.save(img_path)

    print(f"Created {num_gallery} gallery images")

    # Create a few junk images (person_id = -1)
    for i in range(5):
        filename = f"-0001_c{i+1}s1_{i*100:06d}_01.jpg"
        img_path = gallery_dir / filename
        img = Image.fromarray(np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8))
        img.save(img_path)

    print("Created 5 junk images")


def test_filename_parsing():
    """Test filename parsing"""
    print("\n=== Testing Filename Parsing ===")

    # Test valid filename
    metadata = Market1501Dataset.parse_filename("0001_c2s3_000451_03.jpg")
    assert metadata is not None
    assert metadata.person_id == 1
    assert metadata.camera_id == 2
    assert metadata.sequence_id == 3
    assert metadata.frame_id == 451
    assert metadata.detection_id == 3
    assert not metadata.is_distractor
    assert not metadata.is_junk
    print("✓ Valid filename parsed correctly")

    # Test distractor
    metadata = Market1501Dataset.parse_filename("0000_c1s1_000100_01.jpg")
    assert metadata.person_id == 0
    assert metadata.is_distractor
    assert not metadata.is_junk
    print("✓ Distractor filename parsed correctly")

    # Test junk
    metadata = Market1501Dataset.parse_filename("-0001_c1s1_000100_01.jpg")
    assert metadata.person_id == -1
    assert not metadata.is_distractor
    assert metadata.is_junk
    print("✓ Junk filename parsed correctly")

    # Test invalid filename
    metadata = Market1501Dataset.parse_filename("invalid_filename.jpg")
    assert metadata is None
    print("✓ Invalid filename rejected")


def test_dataset_loading():
    """Test dataset loading with synthetic data"""
    print("\n=== Testing Dataset Loading ===")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = Path(temp_dir) / "market1501"
        dataset_path.mkdir()

        # Create synthetic dataset
        create_synthetic_dataset(dataset_path, num_queries=10, num_gallery=50)

        # Load dataset
        dataset = Market1501Dataset(dataset_path)

        # Verify counts
        assert len(dataset.query_metadata) == 10, f"Expected 10 queries, got {len(dataset.query_metadata)}"
        assert len(dataset.gallery_metadata) == 50, f"Expected 50 gallery (junk excluded), got {len(dataset.gallery_metadata)}"
        print(f"✓ Loaded {len(dataset.query_metadata)} queries and {len(dataset.gallery_metadata)} gallery images")

        # Verify ground truth
        assert len(dataset.ground_truth) == 10, f"Expected ground truth for 10 queries, got {len(dataset.ground_truth)}"
        print(f"✓ Ground truth built for {len(dataset.ground_truth)} queries")

        # Check that queries have matches (same person, different camera)
        num_queries_with_matches = sum(1 for indices in dataset.ground_truth.values() if len(indices) > 0)
        print(f"✓ {num_queries_with_matches}/{len(dataset.query_metadata)} queries have matches")

        # Get statistics
        stats = dataset.get_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Queries: {stats['num_queries']}")
        print(f"  Gallery: {stats['num_gallery']}")
        print(f"  Query Identities: {stats['num_query_identities']}")
        print(f"  Gallery Identities: {stats['num_gallery_identities']}")
        print(f"  Avg matches per query: {stats['avg_matches_per_query']:.1f}")

        # Test get_queries()
        queries = dataset.get_queries(with_images=False)
        assert len(queries) == 10
        assert 'person_id' in queries[0]
        assert 'camera_id' in queries[0]
        assert 'file_path' in queries[0]
        print("✓ get_queries() works correctly")

        # Test get_gallery()
        gallery = dataset.get_gallery(with_images=False)
        assert len(gallery) == 50
        print("✓ get_gallery() works correctly")

        # Test image loading
        query = queries[0]
        metadata = dataset.query_metadata[0]
        img = dataset.load_image(metadata)
        assert img.shape == (128, 64, 3), f"Expected (128, 64, 3), got {img.shape}"
        print("✓ Image loading works correctly")

        print(f"\n{dataset}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Market-1501 Dataset Loader Unit Tests")
    print("=" * 60)

    try:
        test_filename_parsing()
        test_dataset_loading()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
