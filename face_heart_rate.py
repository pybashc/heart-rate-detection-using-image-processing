import numpy as np
import cv2
import sys
import time
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

try:
    from cvzone.FaceDetectionModule import FaceDetector
    from cvzone.PlotModule import LivePlot
    import cvzone
except ImportError as e:
    print(f"Error: Missing required dependencies. Install with: pip install cvzone")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Configuration for video capture and processing"""
    real_width: int = 640
    real_height: int = 480
    video_width: int = 160
    video_height: int = 120
    video_channels: int = 3
    frame_rate: int = 15
    
@dataclass
class ProcessingConfig:
    """Configuration for signal processing"""
    pyramid_levels: int = 3
    amplification_alpha: int = 170
    min_frequency: float = 1.0  # Hz (60 BPM)
    max_frequency: float = 2.0  # Hz (120 BPM)
    buffer_size: int = 150
    bpm_calculation_frequency: int = 10
    bpm_buffer_size: int = 10

class HeartRateMonitor:
    """Modern heart rate detection using Eulerian Video Magnification"""
    
    def __init__(self, video_config: VideoConfig = None, processing_config: ProcessingConfig = None):
        self.video_config = video_config or VideoConfig()
        self.processing_config = processing_config or ProcessingConfig()
        
        # Initialize components
        self.webcam: Optional[cv2.VideoCapture] = None
        self.detector = FaceDetector(minDetectionCon=0.7)
        self.plot = LivePlot(
            self.video_config.real_width, 
            self.video_config.real_height, 
            [60, 120], 
            invert=True
        )
        
        # Processing buffers
        self._initialize_buffers()
        
        # Tracking variables
        self.buffer_index = 0
        self.bpm_buffer_index = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        logger.info("Heart Rate Monitor initialized")
    
    def _initialize_buffers(self):
        """Initialize processing buffers and filters"""
        # Create initial Gaussian pyramid
        first_frame = np.zeros((
            self.video_config.video_height, 
            self.video_config.video_width, 
            self.video_config.video_channels
        ))
        first_gauss = self._build_gaussian_pyramid(first_frame, self.processing_config.pyramid_levels + 1)
        target_level = first_gauss[self.processing_config.pyramid_levels]
        
        # Initialize video buffer
        self.video_gauss = np.zeros((
            self.processing_config.buffer_size,
            target_level.shape[0],
            target_level.shape[1],
            self.video_config.video_channels
        ))
        
        # Initialize frequency domain processing
        self.fourier_avg = np.zeros(self.processing_config.buffer_size)
        self.bpm_buffer = np.zeros(self.processing_config.bpm_buffer_size)
        
        # Create bandpass filter mask
        frequencies = (self.video_config.frame_rate * np.arange(self.processing_config.buffer_size) / 
                      self.processing_config.buffer_size)
        self.frequency_mask = ((frequencies >= self.processing_config.min_frequency) & 
                              (frequencies <= self.processing_config.max_frequency))
        self.frequencies = frequencies
        
    def _build_gaussian_pyramid(self, frame: np.ndarray, levels: int) -> List[np.ndarray]:
        """Build Gaussian pyramid for the frame"""
        pyramid = [frame.copy()]
        current_frame = frame.copy()
        
        for _ in range(levels):
            current_frame = cv2.pyrDown(current_frame)
            pyramid.append(current_frame)
        
        return pyramid
    
    def _reconstruct_frame(self, pyramid_frame: np.ndarray, levels: int) -> np.ndarray:
        """Reconstruct frame from Gaussian pyramid level"""
        reconstructed = pyramid_frame.copy()
        
        for _ in range(levels):
            reconstructed = cv2.pyrUp(reconstructed)
        
        # Ensure correct dimensions
        h, w = self.video_config.video_height, self.video_config.video_width
        return reconstructed[:h, :w]
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        return fps
    
    def _process_face_region(self, face_region: np.ndarray) -> Tuple[np.ndarray, float]:
        """Process detected face region for heart rate extraction"""
        # Resize to processing dimensions
        resized_region = cv2.resize(
            face_region, 
            (self.video_config.video_width, self.video_config.video_height)
        )
        
        # Build Gaussian pyramid and store in buffer
        pyramid = self._build_gaussian_pyramid(resized_region, self.processing_config.pyramid_levels + 1)
        self.video_gauss[self.buffer_index] = pyramid[self.processing_config.pyramid_levels]
        
        # Perform FFT on the buffer
        fourier_transform = np.fft.fft(self.video_gauss, axis=0)
        
        # Apply bandpass filter
        fourier_transform[~self.frequency_mask] = 0
        
        # Calculate BPM periodically
        bpm = 0.0
        if self.buffer_index % self.processing_config.bpm_calculation_frequency == 0:
            # Average across spatial dimensions
            for i in range(self.processing_config.buffer_size):
                self.fourier_avg[i] = np.real(fourier_transform[i]).mean()
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(self.fourier_avg)
            hz = self.frequencies[dominant_freq_idx]
            bpm = 60.0 * hz
            
            # Update BPM buffer
            self.bpm_buffer[self.bpm_buffer_index] = bpm
            self.bpm_buffer_index = (self.bpm_buffer_index + 1) % self.processing_config.bpm_buffer_size
        
        # Reconstruct amplified frame
        filtered = np.real(np.fft.ifft(fourier_transform, axis=0))
        filtered *= self.processing_config.amplification_alpha
        
        amplified_frame = self._reconstruct_frame(
            filtered[self.buffer_index], 
            self.processing_config.pyramid_levels
        )
        
        # Combine original and amplified
        output_frame = resized_region + amplified_frame
        output_frame = cv2.convertScaleAbs(output_frame)
        
        # Update buffer index
        self.buffer_index = (self.buffer_index + 1) % self.processing_config.buffer_size
        
        return output_frame, bpm
    
    def _draw_ui(self, frame: np.ndarray, fps: float, bpm: float, face_detected: bool) -> np.ndarray:
        """Draw user interface elements"""
        frame_copy = frame.copy()
        
        # Draw FPS
        cv2.putText(
            frame_copy, f'FPS: {int(fps)}', (30, 440), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        
        # Draw BPM or status
        if face_detected:
            if self.frame_count > self.processing_config.bpm_buffer_size:
                avg_bpm = self.bpm_buffer.mean()
                cvzone.putTextRect(
                    frame_copy, f'BPM: {avg_bpm:.1f}', 
                    (self.video_config.video_width//2, 40), scale=2, 
                    colorR=(0, 255, 0)
                )
            else:
                cvzone.putTextRect(
                    frame_copy, "Calculating BPM...", (30, 40), 
                    scale=2, colorR=(255, 255, 0)
                )
        else:
            cvzone.putTextRect(
                frame_copy, "No face detected", (30, 40), 
                scale=2, colorR=(0, 0, 255)
            )
        
        return frame_copy
    
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """Initialize camera with error handling"""
        try:
            self.webcam = cv2.VideoCapture(camera_index)
            if not self.webcam.isOpened():
                logger.error(f"Failed to open camera {camera_index}")
                return False
            
            # Set camera properties
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_config.real_width)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_config.real_height)
            self.webcam.set(cv2.CAP_PROP_FPS, self.video_config.frame_rate)
            
            logger.info(f"Camera {camera_index} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def run(self, show_video: bool = True) -> None:
        """Main processing loop"""
        if not self.initialize_camera():
            return
        
        logger.info("Starting heart rate monitoring. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = self.webcam.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Calculate FPS
                fps = self._calculate_fps()
                
                # Detect faces
                frame, faces = self.detector.findFaces(frame, draw=False)
                
                if faces:
                    # Process first detected face
                    bbox = faces[0]['bbox']
                    x, y, w, h = bbox
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    
                    # Extract and process face region
                    face_region = frame[y:y + h, x:x + w]
                    processed_frame, bpm = self._process_face_region(face_region)
                    
                    # Display processed region (small overlay)
                    if show_video:
                        small_processed = cv2.resize(
                            processed_frame, 
                            (self.video_config.video_width//2, self.video_config.video_height//2)
                        )
                        frame[0:self.video_config.video_height//2, 
                              -self.video_config.video_width//2:] = small_processed
                    
                    # Update plot
                    if self.frame_count > self.processing_config.bpm_buffer_size:
                        avg_bpm = self.bpm_buffer.mean()
                        plot_img = self.plot.update(float(avg_bpm))
                    else:
                        plot_img = self.plot.update(0)
                    
                    face_detected = True
                else:
                    plot_img = self.plot.update(0)
                    face_detected = False
                
                # Draw UI
                frame_with_ui = self._draw_ui(frame, fps, 0, face_detected)
                
                if show_video:
                    # Stack frames for display
                    display_frame = cvzone.stackImages([frame_with_ui, plot_img], 2, 1)
                    cv2.imshow("Heart Rate Monitor", display_frame)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.webcam:
            self.webcam.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

def main():
    """Main entry point"""
    # Create custom configurations if needed
    video_config = VideoConfig(
        real_width=640,
        real_height=480,
        frame_rate=30  # Higher frame rate for better accuracy
    )
    
    processing_config = ProcessingConfig(
        min_frequency=0.8,  # 48 BPM
        max_frequency=3.0,  # 180 BPM
        amplification_alpha=150
    )
    
    # Initialize and run monitor
    monitor = HeartRateMonitor(video_config, processing_config)
    
    try:
        monitor.run(show_video=len(sys.argv) == 1)  # Show video unless argument provided
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
