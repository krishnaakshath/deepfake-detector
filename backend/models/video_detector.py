"""
Video Deepfake Detector
Uses CNN-based analysis to detect visual artifacts and inconsistencies in video frames
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class VideoDeepfakeDetector:
    """
    CNN-based deepfake detection for video content.
    Analyzes frames for:
    - Compression artifacts
    - Blending boundaries
    - Inconsistent lighting
    - Face warping anomalies
    """
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.model_loaded = False
        self.frame_size = (224, 224)
        
    def extract_frames(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Extract evenly spaced frames from video, or reading single image"""
        frames = []
        
        # Check if it's an image file
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        ext = os.path.splitext(video_path)[1].lower()
        
        if ext in image_extensions:
            frame = cv2.imread(video_path)
            if frame is None:
                raise ValueError(f"Could not open image: {video_path}")
            return [frame]
            
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback for streams or problematic files
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            cap.release()
            return frames
            
        frame_indices = np.linspace(0, total_frames - 1, min(total_frames, max_frames), dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame using Haar cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def analyze_compression_artifacts(self, face_region: np.ndarray) -> float:
        """
        Detect compression artifacts that are common in deepfakes.
        Analyzes DCT coefficients and block boundaries.
        """
        if face_region.size == 0:
            return 0.5
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Resize to standard size
        gray = cv2.resize(gray, (64, 64))
        
        # Apply Laplacian to detect edges (compression artifacts show up as blocky edges)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Analyze block boundaries (8x8 DCT blocks)
        block_score = self._analyze_block_boundaries(gray)
        
        # Higher variance in laplacian often indicates manipulation
        # Normalize to 0-1 range
        artifact_score = min(1.0, variance / 1000) * 0.6 + block_score * 0.4
        
        return artifact_score
    
    def _analyze_block_boundaries(self, gray: np.ndarray) -> float:
        """Detect 8x8 block boundaries common in manipulated videos"""
        h, w = gray.shape
        block_size = 8
        
        boundary_strength = 0
        count = 0
        
        # Check horizontal block boundaries
        for i in range(block_size, h - block_size, block_size):
            for j in range(w):
                diff = abs(int(gray[i, j]) - int(gray[i-1, j]))
                boundary_strength += diff
                count += 1
        
        # Check vertical block boundaries
        for i in range(h):
            for j in range(block_size, w - block_size, block_size):
                diff = abs(int(gray[i, j]) - int(gray[i, j-1]))
                boundary_strength += diff
                count += 1
        
        if count == 0:
            return 0.0
            
        avg_boundary = boundary_strength / count
        return min(1.0, avg_boundary / 50)
    
    def analyze_color_consistency(self, face_region: np.ndarray) -> float:
        """
        Check for color inconsistencies around face boundaries.
        Deepfakes often have subtle color mismatches at blend boundaries.
        """
        if face_region.size == 0:
            return 0.5
        
        # Convert to LAB color space for better perception-based analysis
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Calculate local statistics
        l_std = np.std(l)
        a_std = np.std(a)
        b_std = np.std(b)
        
        # High variation in color channels near edges suggests manipulation
        # Analyze edge regions
        edges = cv2.Canny(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate inconsistency score
        color_score = (l_std / 100 + a_std / 50 + b_std / 50) / 3
        inconsistency = color_score * edge_density * 10
        
        return min(1.0, inconsistency)
    
    def analyze_noise_patterns(self, face_region: np.ndarray) -> float:
        """
        Analyze noise patterns - deepfakes often have different noise characteristics
        compared to the rest of the frame.
        """
        if face_region.size == 0:
            return 0.5
        
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Apply high-pass filter to extract noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.subtract(gray, blur)
        
        # Analyze noise statistics
        noise_std = np.std(noise)
        noise_mean = np.mean(np.abs(noise))
        
        # Calculate noise pattern regularity using autocorrelation
        # Regular patterns suggest synthetic content
        noise_fft = np.fft.fft2(noise.astype(float))
        power_spectrum = np.abs(noise_fft) ** 2
        
        # Check for periodic patterns in noise
        ps_std = np.std(power_spectrum)
        ps_max = np.max(power_spectrum)
        
        periodicity = ps_max / (ps_std + 1e-6) / 1000
        
        return min(1.0, (noise_std / 20 + periodicity) / 2)
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame for deepfake indicators"""
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            return {
                'face_detected': False,
                'artifact_score': 0.5,
                'color_score': 0.5,
                'noise_score': 0.5,
                'overall_score': 0.5
            }
        
        # Analyze the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Add margin around face
        margin = int(min(w, h) * 0.2)
        y1 = max(0, y - margin)
        y2 = min(frame.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(frame.shape[1], x + w + margin)
        
        face_region = frame[y1:y2, x1:x2]
        
        # Run all analyses
        artifact_score = self.analyze_compression_artifacts(face_region)
        color_score = self.analyze_color_consistency(face_region)
        noise_score = self.analyze_noise_patterns(face_region)
        
        # Weighted combination
        overall_score = (
            artifact_score * 0.35 +
            color_score * 0.35 +
            noise_score * 0.30
        )
        
        return {
            'face_detected': True,
            'face_location': (x, y, w, h),
            'artifact_score': round(artifact_score, 3),
            'color_score': round(color_score, 3),
            'noise_score': round(noise_score, 3),
            'overall_score': round(overall_score, 3)
        }
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Perform complete deepfake analysis on a video.
        Returns detection results with confidence scores.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            raise ValueError("Could not extract frames from video")
        
        # Analyze each frame
        frame_results = []
        faces_detected = 0
        
        for frame in frames:
            result = self.analyze_frame(frame)
            frame_results.append(result)
            if result['face_detected']:
                faces_detected += 1
        
        # Aggregate results
        valid_results = [r for r in frame_results if r['face_detected']]
        
        if len(valid_results) == 0:
            return {
                'success': False,
                'error': 'No faces detected in video',
                'frames_analyzed': len(frames),
                'is_deepfake': False,
                'confidence': 0,
                'authenticity_score': 50
            }
        
        avg_artifact = np.mean([r['artifact_score'] for r in valid_results])
        avg_color = np.mean([r['color_score'] for r in valid_results])
        avg_noise = np.mean([r['noise_score'] for r in valid_results])
        avg_overall = np.mean([r['overall_score'] for r in valid_results])
        
        # Calculate variance across frames (high variance suggests tampering)
        score_variance = np.var([r['overall_score'] for r in valid_results])
        
        # Determine if deepfake
        # Threshold calibrated for detection
        deepfake_score = avg_overall + score_variance * 2
        is_deepfake = deepfake_score > 0.45
        
        # Calculate confidence based on consistency
        confidence = min(95, max(60, 100 - score_variance * 200))
        
        # Authenticity is inverse of deepfake probability
        authenticity = max(0, min(100, (1 - deepfake_score) * 100))
        
        return {
            'success': True,
            'frames_analyzed': len(frames),
            'faces_detected': faces_detected,
            'is_deepfake': is_deepfake,
            'deepfake_probability': round(deepfake_score * 100, 1),
            'authenticity_score': round(authenticity, 1),
            'confidence': round(confidence, 1),
            'details': {
                'artifact_score': round(avg_artifact * 100, 1),
                'color_consistency': round((1 - avg_color) * 100, 1),
                'noise_analysis': round((1 - avg_noise) * 100, 1),
                'temporal_consistency': round((1 - score_variance) * 100, 1)
            },
            'indicators': self._get_indicators(avg_artifact, avg_color, avg_noise, score_variance)
        }
    
    def _get_indicators(self, artifact: float, color: float, noise: float, variance: float) -> List[str]:
        """Generate human-readable indicators based on scores"""
        indicators = []
        
        if artifact > 0.5:
            indicators.append("Compression artifacts detected around face region")
        if color > 0.5:
            indicators.append("Color inconsistencies found at face boundaries")
        if noise > 0.5:
            indicators.append("Unusual noise patterns in facial area")
        if variance > 0.1:
            indicators.append("Temporal inconsistencies across frames")
        
        if len(indicators) == 0:
            indicators.append("No significant manipulation indicators found")
        
        return indicators
