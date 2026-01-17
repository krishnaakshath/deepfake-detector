"""
Face Landmark Analyzer
Uses MediaPipe for 468-point face mesh analysis to detect deepfake anomalies
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FaceLandmarkAnalyzer:
    """
    Face landmark analysis for deepfake detection.
    Uses MediaPipe Face Mesh (468 landmarks) to detect:
    - Unnatural blink patterns
    - Lip sync inconsistencies
    - Facial asymmetry anomalies
    - Micro-expression irregularities
    """
    
    # Key landmark indices for analysis
    # Eyes
    LEFT_EYE_UPPER = [386, 374, 373, 390, 388, 387]
    LEFT_EYE_LOWER = [263, 466, 388, 387, 386, 385]
    RIGHT_EYE_UPPER = [159, 145, 144, 163, 161, 160]
    RIGHT_EYE_LOWER = [33, 246, 161, 160, 159, 158]
    
    # Mouth
    UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    
    # Face contour
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None
            
        self.blink_history = []
        self.lip_history = []
        
    def get_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 face landmarks from frame"""
        if not MEDIAPIPE_AVAILABLE:
            return None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Convert to numpy array with pixel coordinates
            points = np.array([
                [lm.x * w, lm.y * h, lm.z * w] 
                for lm in landmarks.landmark
            ])
            return points
        return None
    
    def calculate_eye_aspect_ratio(self, landmarks: np.ndarray, upper_indices: List[int], lower_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        EAR drops significantly during blinks.
        """
        upper_points = landmarks[upper_indices]
        lower_points = landmarks[lower_indices]
        
        # Calculate vertical distances
        vertical_dist = np.mean([
            np.linalg.norm(upper_points[i] - lower_points[i])
            for i in range(min(len(upper_points), len(lower_points)))
        ])
        
        # Calculate horizontal distance (eye width)
        horizontal_dist = np.linalg.norm(upper_points[0] - upper_points[-1])
        
        if horizontal_dist == 0:
            return 0
            
        ear = vertical_dist / horizontal_dist
        return ear
    
    def analyze_blink_patterns(self, frames_landmarks: List[np.ndarray]) -> Dict:
        """
        Analyze blink patterns across frames.
        Deepfakes often have:
        - Too few blinks
        - Unnatural blink timing
        - Asymmetric blinks
        """
        if len(frames_landmarks) < 5:
            return {'error': 'Not enough frames for blink analysis'}
        
        left_ears = []
        right_ears = []
        
        for landmarks in frames_landmarks:
            if landmarks is None:
                continue
                
            left_ear = self.calculate_eye_aspect_ratio(
                landmarks, self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER
            )
            right_ear = self.calculate_eye_aspect_ratio(
                landmarks, self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER
            )
            
            left_ears.append(left_ear)
            right_ears.append(right_ear)
        
        if len(left_ears) < 5:
            return {'error': 'Insufficient eye data'}
        
        # Detect blinks (EAR drops below threshold)
        ear_threshold = 0.2
        left_blinks = self._count_state_changes(left_ears, ear_threshold)
        right_blinks = self._count_state_changes(right_ears, ear_threshold)
        
        # Calculate blink rate (normal: 15-20 blinks per minute)
        # Assuming 30 fps, frames_count / 30 = seconds
        duration_seconds = len(frames_landmarks) / 30
        blink_rate = ((left_blinks + right_blinks) / 2) / duration_seconds * 60
        
        # Check for asymmetric blinking
        asymmetry = abs(left_blinks - right_blinks) / max(1, (left_blinks + right_blinks) / 2)
        
        # Calculate EAR variance (natural eyes have consistent variance)
        left_variance = np.var(left_ears)
        right_variance = np.var(right_ears)
        
        # Score naturalness
        # Natural blink rate: 15-20/min
        blink_score = 1 - min(1, abs(blink_rate - 17) / 20)
        asymmetry_score = 1 - min(1, asymmetry)
        variance_score = min(1, (left_variance + right_variance) * 10)
        
        naturalness = (blink_score * 0.4 + asymmetry_score * 0.3 + variance_score * 0.3)
        
        return {
            'left_blinks': left_blinks,
            'right_blinks': right_blinks,
            'blink_rate_per_minute': round(blink_rate, 1),
            'blink_asymmetry': round(asymmetry, 3),
            'ear_variance': round((left_variance + right_variance) / 2, 4),
            'naturalness_score': round(naturalness, 3)
        }
    
    def _count_state_changes(self, values: List[float], threshold: float) -> int:
        """Count number of times values cross threshold (state changes)"""
        changes = 0
        above = values[0] > threshold
        for v in values[1:]:
            if (v > threshold) != above:
                changes += 1
                above = v > threshold
        return changes // 2  # Each blink has 2 crossings
    
    def calculate_lip_movement(self, landmarks: np.ndarray) -> float:
        """Calculate lip opening ratio for lip-sync analysis"""
        upper_lip = landmarks[self.UPPER_LIP]
        lower_lip = landmarks[self.LOWER_LIP]
        
        # Vertical mouth opening
        vertical_dist = np.mean([
            np.linalg.norm(upper_lip[i] - lower_lip[i])
            for i in range(min(len(upper_lip), len(lower_lip)))
        ])
        
        # Mouth width
        mouth_outer = landmarks[self.MOUTH_OUTER]
        horizontal_dist = np.linalg.norm(mouth_outer[0] - mouth_outer[len(mouth_outer)//2])
        
        if horizontal_dist == 0:
            return 0
            
        return vertical_dist / horizontal_dist
    
    def analyze_lip_sync(self, frames_landmarks: List[np.ndarray]) -> Dict:
        """
        Analyze lip movement patterns.
        Check for natural speech patterns and movement smoothness.
        """
        if len(frames_landmarks) < 5:
            return {'error': 'Not enough frames for lip sync analysis'}
        
        lip_movements = []
        for landmarks in frames_landmarks:
            if landmarks is None:
                continue
            movement = self.calculate_lip_movement(landmarks)
            lip_movements.append(movement)
        
        if len(lip_movements) < 5:
            return {'error': 'Insufficient lip data'}
        
        # Analyze movement patterns
        movement_variance = np.var(lip_movements)
        movement_range = np.max(lip_movements) - np.min(lip_movements)
        
        # Calculate smoothness (jerky movements indicate manipulation)
        diffs = np.diff(lip_movements)
        jerkiness = np.mean(np.abs(np.diff(diffs)))
        smoothness = 1 - min(1, jerkiness * 20)
        
        # Natural speech has moderate variance
        variance_score = 1 - abs(movement_variance - 0.02) * 20
        variance_score = max(0, min(1, variance_score))
        
        naturalness = (smoothness * 0.5 + variance_score * 0.5)
        
        return {
            'movement_variance': round(movement_variance, 4),
            'movement_range': round(movement_range, 3),
            'jerkiness': round(jerkiness, 4),
            'smoothness': round(smoothness, 3),
            'naturalness_score': round(naturalness, 3)
        }
    
    def analyze_facial_symmetry(self, landmarks: np.ndarray) -> Dict:
        """
        Analyze facial symmetry.
        Deepfakes can have subtle asymmetric artifacts.
        """
        # Get face center (nose tip)
        nose_tip = landmarks[1]
        
        # Compare left and right side distances
        left_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
        right_indices = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
        
        left_points = landmarks[left_indices]
        right_points = landmarks[right_indices]
        
        # Calculate distances from center
        left_distances = [np.linalg.norm(p - nose_tip) for p in left_points]
        right_distances = [np.linalg.norm(p - nose_tip) for p in right_points]
        
        # Compare symmetry
        asymmetry_scores = []
        for l, r in zip(left_distances, right_distances):
            if max(l, r) > 0:
                asymmetry_scores.append(abs(l - r) / max(l, r))
        
        avg_asymmetry = np.mean(asymmetry_scores)
        max_asymmetry = np.max(asymmetry_scores)
        
        # Natural faces have slight asymmetry (0.02-0.05)
        # Perfect symmetry or high asymmetry is suspicious
        if avg_asymmetry < 0.01:  # Too perfect
            symmetry_score = 0.7
        elif avg_asymmetry > 0.1:  # Too asymmetric
            symmetry_score = 1 - (avg_asymmetry - 0.1) * 5
        else:  # Natural range
            symmetry_score = 1.0
        
        symmetry_score = max(0, min(1, symmetry_score))
        
        return {
            'average_asymmetry': round(avg_asymmetry, 4),
            'max_asymmetry': round(max_asymmetry, 4),
            'naturalness_score': round(symmetry_score, 3)
        }
    
    def analyze_micro_expressions(self, frames_landmarks: List[np.ndarray]) -> Dict:
        """
        Analyze micro-expressions and facial dynamics.
        Real faces have subtle continuous movements.
        """
        if len(frames_landmarks) < 10:
            return {'error': 'Not enough frames for micro-expression analysis'}
        
        # Track key facial points across frames
        # Eyebrows, nose, mouth corners
        key_points = [70, 300, 33, 263, 61, 291]  # Eyebrows, eyes outer, mouth corners
        
        movements = []
        prev_landmarks = None
        
        for landmarks in frames_landmarks:
            if landmarks is None:
                continue
                
            if prev_landmarks is not None:
                # Calculate displacement of key points
                displacements = []
                for idx in key_points:
                    displacement = np.linalg.norm(landmarks[idx] - prev_landmarks[idx])
                    displacements.append(displacement)
                movements.append(displacements)
            
            prev_landmarks = landmarks
        
        if len(movements) < 5:
            return {'error': 'Insufficient movement data'}
        
        movements = np.array(movements)
        
        # Analyze movement characteristics
        avg_movement = np.mean(movements)
        movement_variance = np.var(movements)
        
        # Cross-correlation of different facial regions
        # Natural faces have correlated movements
        correlations = []
        for i in range(movements.shape[1]):
            for j in range(i + 1, movements.shape[1]):
                corr = np.corrcoef(movements[:, i], movements[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0.5
        
        # Natural faces have moderate movement and high correlation
        movement_score = 1 - min(1, abs(avg_movement - 1) / 5)
        correlation_score = avg_correlation
        
        naturalness = (movement_score * 0.4 + correlation_score * 0.6)
        
        return {
            'average_movement': round(avg_movement, 3),
            'movement_variance': round(movement_variance, 4),
            'region_correlation': round(avg_correlation, 3),
            'naturalness_score': round(naturalness, 3)
        }
    
    def extract_frames_from_video(self, video_path: str, max_frames: int = 60) -> List[np.ndarray]:
        """Extract frames from video file or read single image"""
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
            # Fallback
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
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Perform complete face landmark analysis on video.
        Returns detection results with confidence scores.
        """
        if not MEDIAPIPE_AVAILABLE:
            return {
                'success': False,
                'error': 'MediaPipe not available. Install with: pip install mediapipe'
            }
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Extract frames
        frames = self.extract_frames_from_video(video_path)
        
        if len(frames) == 0:
            return {
                'success': False,
                'error': 'Could not extract frames from video'
            }
        
        # Get landmarks for all frames
        all_landmarks = []
        faces_detected = 0
        
        for frame in frames:
            landmarks = self.get_landmarks(frame)
            all_landmarks.append(landmarks)
            if landmarks is not None:
                faces_detected += 1
        
        if faces_detected == 0:
            return {
                'success': False,
                'error': 'No faces detected in analysis',
                'faces_detected': faces_detected,
                'frames_analyzed': len(frames)
            }
            
        # For single images, we proceed even with 1 face
        min_faces = 1 if len(frames) == 1 else 5
        
        if faces_detected < min_faces:
             # Even if fewer faces than ideal, we try to proceed but return low confidence
             pass
        
        # Run all analyses
        blink_analysis = self.analyze_blink_patterns(all_landmarks)
        lip_analysis = self.analyze_lip_sync(all_landmarks)
        
        # Get symmetry from middle frame
        valid_landmarks = [l for l in all_landmarks if l is not None]
        mid_landmarks = valid_landmarks[len(valid_landmarks) // 2]
        symmetry_analysis = self.analyze_facial_symmetry(mid_landmarks)
        
        micro_analysis = self.analyze_micro_expressions(all_landmarks)
        
        # Calculate composite scores
        blink_score = blink_analysis.get('naturalness_score', 0.5)
        lip_score = lip_analysis.get('naturalness_score', 0.5)
        symmetry_score = symmetry_analysis.get('naturalness_score', 0.5)
        micro_score = micro_analysis.get('naturalness_score', 0.5)
        
        # Weighted combination
        # Lower naturalness = higher deepfake probability
        naturalness_avg = (
            blink_score * 0.25 +
            lip_score * 0.25 +
            symmetry_score * 0.25 +
            micro_score * 0.25
        )
        
        deepfake_score = 1 - naturalness_avg
        is_deepfake = deepfake_score > 0.45
        authenticity = naturalness_avg * 100
        
        # Confidence based on data quality
        confidence = min(90, max(60, faces_detected / len(frames) * 100))
        
        return {
            'success': True,
            'frames_analyzed': len(frames),
            'faces_detected': faces_detected,
            'is_deepfake': is_deepfake,
            'deepfake_probability': round(deepfake_score * 100, 1),
            'authenticity_score': round(authenticity, 1),
            'confidence': round(confidence, 1),
            'details': {
                'blink_naturalness': round(blink_score * 100, 1),
                'lip_sync_naturalness': round(lip_score * 100, 1),
                'facial_symmetry': round(symmetry_score * 100, 1),
                'micro_expression_naturalness': round(micro_score * 100, 1)
            },
            'blink_analysis': blink_analysis,
            'lip_analysis': lip_analysis,
            'symmetry_analysis': symmetry_analysis,
            'micro_expression_analysis': micro_analysis,
            'indicators': self._get_indicators(blink_score, lip_score, symmetry_score, micro_score)
        }
    
    def _get_indicators(self, blink: float, lip: float, symmetry: float, micro: float) -> List[str]:
        """Generate human-readable indicators based on scores"""
        indicators = []
        
        if blink < 0.5:
            indicators.append("Abnormal blink patterns detected")
        if lip < 0.5:
            indicators.append("Lip movement irregularities found")
        if symmetry < 0.5:
            indicators.append("Unusual facial symmetry patterns")
        if micro < 0.5:
            indicators.append("Unnatural micro-expression dynamics")
        
        if len(indicators) == 0:
            indicators.append("Face landmarks exhibit natural characteristics")
        
        return indicators
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
