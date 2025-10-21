#!/usr/bin/env python3
"""
Video Dimensions Diagnostic Tool
Checks your input videos and calculates expected grid dimensions
"""
import cv2
import sys

def check_video_dimensions(video_paths):
    """Check dimensions of input videos"""
    print("="*60)
    print("VIDEO DIMENSIONS DIAGNOSTIC")
    print("="*60)
    
    dimensions = []
    for i, path in enumerate(video_paths, 1):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ Camera {i}: Failed to open {path}")
            continue
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        dimensions.append((w, h))
        print(f"\n📹 Camera {i}: {path}")
        print(f"   Resolution: {w}x{h}")
        print(f"   FPS: {fps:.1f}")
        print(f"   Frames: {frame_count}")
        
        cap.release()
    
    print("\n" + "="*60)
    print("GRID CALCULATIONS")
    print("="*60)
    
    if dimensions:
        # Check if all videos have same dimensions
        if len(set(dimensions)) == 1:
            w, h = dimensions[0]
            grid_w = w * 2
            grid_h = h * 2
            print(f"\n✅ All videos have same resolution: {w}x{h}")
            print(f"📐 Expected 2x2 grid dimensions: {grid_w}x{grid_h}")
        else:
            print(f"\n⚠️  Videos have DIFFERENT resolutions:")
            for i, (w, h) in enumerate(dimensions, 1):
                print(f"   Camera {i}: {w}x{h}")
            
            # Grid will use the first video's dimensions
            target_w, target_h = dimensions[0]
            grid_w = target_w * 2
            grid_h = target_h * 2
            print(f"\n📐 Grid will resize all to: {target_w}x{target_h} (from Camera 1)")
            print(f"📐 Final 2x2 grid dimensions: {grid_w}x{grid_h}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if dimensions:
        target_w, target_h = dimensions[0]
        grid_w = target_w * 2
        grid_h = target_h * 2
        
        print(f"\nThe VideoWriter will be initialized with: {grid_w}x{grid_h}")
        print(f"All frames must match these EXACT dimensions.")
        print(f"\n✅ The fixed pipeline automatically handles this!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Default test videos
    video_paths = [
        'test_the_pipeline/test_videos/test_video_1.mp4',
        'test_the_pipeline/test_videos/test_video_2.mp4',
        'test_the_pipeline/test_videos/test_video_3.mp4',
        'test_the_pipeline/test_videos/test_video_4.mp4'
    ]
    
    # Override with command line args if provided
    if len(sys.argv) > 1:
        video_paths = sys.argv[1:5]  # Take first 4 arguments
    
    check_video_dimensions(video_paths)