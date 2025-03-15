import unittest
import numpy as np
import cv2
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import FastDepthEstimator

class TestDepthEstimator(unittest.TestCase):
    def setUp(self):
        # Create a small test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a gradient for depth testing
        for i in range(100):
            self.test_image[:, i, :] = i * 2
        
        # Initialize with CPU to ensure tests run anywhere
        self.estimator = FastDepthEstimator(device="cpu")
    
    def tearDown(self):
        # Clean up resources
        self.estimator.release_resources()
    
    def test_estimate_depth(self):
        # Basic test to ensure the function runs without error
        try:
            depth_map = self.estimator.estimate_depth(self.test_image)
            # Check that we get a valid depth map
            self.assertIsNotNone(depth_map)
            self.assertEqual(depth_map.shape[:2], self.test_image.shape[:2])
            self.assertEqual(depth_map.dtype, np.uint8)
        except Exception as e:
            self.fail(f"estimate_depth raised {type(e).__name__} unexpectedly: {str(e)}")
    
    def test_batch_estimate_depth(self):
        # Test batch processing with multiple identical images
        batch = [self.test_image] * 2
        try:
            depth_maps = self.estimator.batch_estimate_depth(batch)
            # Check that we get valid depth maps
            self.assertEqual(len(depth_maps), len(batch))
            for depth_map in depth_maps:
                self.assertEqual(depth_map.shape[:2], self.test_image.shape[:2])
                self.assertEqual(depth_map.dtype, np.uint8)
        except Exception as e:
            self.fail(f"batch_estimate_depth raised {type(e).__name__} unexpectedly: {str(e)}")

if __name__ == '__main__':
    unittest.main()
