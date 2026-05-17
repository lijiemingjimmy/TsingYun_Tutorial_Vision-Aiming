import sys
import cv2
from simulator_client.pipeline import FallbackPipeline
from simulator_client.protocol import Matrix3x3
from detector import detect_mnist_board

pipeline = FallbackPipeline()

def mock_process(self, image, camera_matrix, timestamp=0.0):
    detections = detect_mnist_board(image, threshold=self.threshold)
    good = [d for d in detections if d.class_id >= 0 and d.confidence >= 0.2]
    print(f"Time {timestamp:.2f}: Detections: {[(d.class_id, d.confidence) for d in detections]}")
    print(f"Time {timestamp:.2f}: Good: {[(d.class_id, d.confidence) for d in good]}")
    return self._center_fallback(image, camera_matrix, "debug")

FallbackPipeline.process_rgb_image = mock_process

if __name__ == "__main__":
    import simulator.runner as runner
    runner.main()
