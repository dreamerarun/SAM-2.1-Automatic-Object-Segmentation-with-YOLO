import cv2
import numpy as np
from ultralytics import SAM, YOLO
import tempfile
import os
import shutil

class AutomaticSAM2Segmenter:
    def __init__(self, sam_model="sam2_t.pt", yolo_model="yolov8n.pt"):
        """
        Initialize automatic SAM2 segmentation using YOLO for object detection
        
        Args:
            sam_model: Path to SAM2 checkpoint
            yolo_model: Path to YOLO model for automatic detection
        """
        print(f"Loading models...")
        print(f"  - SAM2: {sam_model}")
        print(f"  - YOLO: {yolo_model}")
        
        # Initialize models
        self.sam = SAM(sam_model)
        self.yolo = YOLO(yolo_model)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Parameters
        self.frame_width = 1280
        self.frame_height = 720
        self.confidence_threshold = 0.5
        
        # Segmentation mode
        self.mode = "auto"  # "auto" or "everything"
        
        print("Models loaded successfully!")
    
    def segment_everything(self, frame):
        """Segment everything in the frame (no detection needed)"""
        try:
            # Save frame temporarily
            temp_frame_path = os.path.join(self.temp_dir, "temp_frame.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            # Run SAM2 in automatic mode (segment everything)
            results = self.sam(temp_frame_path)
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'masks') and result.masks is not None:
                    # Create overlay for all masks
                    overlay = frame.copy()
                    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    
                    # Generate random colors for each mask
                    num_masks = len(result.masks.data)
                    colors = np.random.randint(0, 255, (num_masks, 3), dtype=np.uint8)
                    
                    for idx, mask in enumerate(result.masks.data):
                        mask_np = mask.cpu().numpy()
                        
                        # Resize mask
                        mask_resized = cv2.resize(
                            mask_np.astype(np.float32),
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        # Apply colored overlay
                        color = colors[idx].tolist()
                        overlay[mask_binary == 1] = color
                        combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
                    
                    # Blend with original
                    result_frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
                    
                    # Draw contours
                    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(result_frame, contours, -1, (255, 255, 255), 1)
                    
                    # Cleanup
                    os.remove(temp_frame_path)
                    
                    return result_frame, num_masks
            
            # Cleanup
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            
            return frame, 0
            
        except Exception as e:
            print(f"Error in segment_everything: {e}")
            return frame, 0
    
    def segment_detected_objects(self, frame):
        """Detect objects with YOLO and segment them with SAM2"""
        try:
            # Run YOLO detection
            yolo_results = self.yolo(frame, conf=self.confidence_threshold, verbose=False)
            
            if not yolo_results or len(yolo_results) == 0:
                return frame, 0
            
            yolo_result = yolo_results[0]
            
            if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
                return frame, 0
            
            # Save frame temporarily
            temp_frame_path = os.path.join(self.temp_dir, "temp_frame.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            # Get bounding boxes
            boxes = yolo_result.boxes.xyxy.cpu().numpy()
            
            # Create overlay
            overlay = frame.copy()
            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Generate random colors
            colors = np.random.randint(0, 255, (len(boxes), 3), dtype=np.uint8)
            
            # Segment each detected object
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                
                # Calculate center point of bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Run SAM2 with center point
                sam_results = self.sam(temp_frame_path, points=[[center_x, center_y]], labels=[1])
                
                if sam_results and len(sam_results) > 0:
                    sam_result = sam_results[0]
                    
                    if hasattr(sam_result, 'masks') and sam_result.masks is not None:
                        mask = sam_result.masks.data[0].cpu().numpy()
                        
                        # Resize mask
                        mask_resized = cv2.resize(
                            mask.astype(np.float32),
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        # Apply colored overlay
                        color = colors[idx].tolist()
                        overlay[mask_binary == 1] = color
                        combined_mask = cv2.bitwise_or(combined_mask, mask_binary)
                        
                        # Draw bounding box
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        
                        # Draw center point
                        cv2.circle(overlay, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Blend with original
            result_frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result_frame, contours, -1, (255, 255, 255), 2)
            
            # Cleanup
            os.remove(temp_frame_path)
            
            return result_frame, len(boxes)
            
        except Exception as e:
            print(f"Error in segment_detected_objects: {e}")
            return frame, 0
    
    def cleanup(self):
        """Cleanup temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up: {e}")
    
    def run(self):
        """Main loop for automatic segmentation"""
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Get actual resolution
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("\n=== SAM2 Automatic Segmentation ===")
        print("Commands:")
        print("  A - Auto mode (YOLO detection + SAM2)")
        print("  E - Everything mode (SAM2 segment all)")
        print("  +/- - Adjust confidence threshold")
        print("  S - Save current frame")
        print("  Q - Quit")
        print("======================================\n")
        print(f"Starting in '{self.mode}' mode...")
        
        window_name = 'SAM2 Automatic Segmentation'
        cv2.namedWindow(window_name)
        
        fps_time = cv2.getTickCount()
        fps = 0
        frame_count = 0
        process_every_n_frames = 3  # Process every 3rd frame for better FPS
        
        last_result = None
        num_objects = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame")
                break
            
            frame_count += 1
            
            # Process frame
            if frame_count % process_every_n_frames == 0:
                if self.mode == "auto":
                    display_frame, num_objects = self.segment_detected_objects(frame)
                else:  # everything mode
                    display_frame, num_objects = self.segment_everything(frame)
                
                last_result = display_frame
            else:
                # Use last result to maintain FPS
                display_frame = last_result if last_result is not None else frame
            
            # Calculate FPS
            if frame_count % 10 == 0:
                current_time = cv2.getTickCount()
                time_diff = (current_time - fps_time) / cv2.getTickFrequency()
                if time_diff > 0:
                    fps = 10.0 / time_diff
                fps_time = current_time
            
            # Display info
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Mode: {self.mode.upper()}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Objects: {num_objects}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if self.mode == "auto":
                cv2.putText(display_frame, f"Confidence: {self.confidence_threshold:.2f}", (10, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a') or key == ord('A'):
                self.mode = "auto"
                print("Switched to AUTO mode (YOLO + SAM2)")
            elif key == ord('e') or key == ord('E'):
                self.mode = "everything"
                print("Switched to EVERYTHING mode (SAM2 only)")
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")
            elif key == ord('s') or key == ord('S'):
                filename = f"segmented_{cv2.getTickCount()}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Saved frame as: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.cleanup()
        print("\nWebcam released. Goodbye!")


if __name__ == "__main__":
    # Initialize automatic segmenter
    # SAM2 models: sam2_t.pt (tiny), sam2_s.pt (small), sam2_b.pt (base), sam2_l.pt (large)
    # YOLO models: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    segmenter = AutomaticSAM2Segmenter(
        sam_model="sam2_t.pt",
        yolo_model="yolov8n.pt"
    )
    segmenter.run()
