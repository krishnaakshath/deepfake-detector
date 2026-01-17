"""
Helper utilities for deepfake detection
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import numpy as np


def convert_to_serializable(obj: Any) -> Any:
    """
    recursively convert numpy types to native python types for json serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    return obj


def generate_report(
    video_result: Optional[Dict] = None,
    audio_result: Optional[Dict] = None,
    face_result: Optional[Dict] = None,
    filename: str = "unknown"
) -> Dict:
    """
    Generate a comprehensive detection report combining all analyses.
    
    Args:
        video_result: Results from video deepfake detection
        audio_result: Results from audio deepfake detection
        face_result: Results from face landmark analysis
        filename: Original filename
    
    Returns:
        Complete analysis report
    """
    report = {
        'filename': filename,
        'timestamp': datetime.utcnow().isoformat(),
        'analyses_performed': [],
        'overall_result': {},
        'details': {}
    }
    
    scores = []
    confidences = []
    all_indicators = []
    
    # Process video results
    if video_result and video_result.get('success'):
        report['analyses_performed'].append('video')
        report['details']['video'] = video_result
        scores.append(video_result.get('authenticity_score', 50))
        confidences.append(video_result.get('confidence', 50))
        all_indicators.extend(video_result.get('indicators', []))
    
    # Process audio results
    if audio_result and audio_result.get('success'):
        report['analyses_performed'].append('audio')
        report['details']['audio'] = audio_result
        scores.append(audio_result.get('authenticity_score', 50))
        confidences.append(audio_result.get('confidence', 50))
        all_indicators.extend(audio_result.get('indicators', []))
    
    # Process face landmark results
    if face_result and face_result.get('success'):
        report['analyses_performed'].append('face_landmarks')
        report['details']['face_landmarks'] = face_result
        scores.append(face_result.get('authenticity_score', 50))
        confidences.append(face_result.get('confidence', 50))
        all_indicators.extend(face_result.get('indicators', []))
    
    # Calculate overall scores
    if scores:
        overall_authenticity = sum(scores) / len(scores)
        overall_confidence = sum(confidences) / len(confidences)
        
        # Determine verdict
        if overall_authenticity >= 70:
            verdict = 'likely_authentic'
            verdict_label = 'Likely Authentic'
            risk_level = 'low'
        elif overall_authenticity >= 50:
            verdict = 'uncertain'
            verdict_label = 'Uncertain - Review Recommended'
            risk_level = 'medium'
        else:
            verdict = 'likely_fake'
            verdict_label = 'Likely Deepfake'
            risk_level = 'high'
        
        report['overall_result'] = {
            'authenticity_score': round(overall_authenticity, 1),
            'confidence': round(overall_confidence, 1),
            'verdict': verdict,
            'verdict_label': verdict_label,
            'risk_level': risk_level,
            'is_deepfake': overall_authenticity < 50
        }
        
        # Deduplicate indicators
        report['indicators'] = list(set(all_indicators))
    else:
        report['overall_result'] = {
            'error': 'No successful analyses completed',
            'verdict': 'unknown',
            'verdict_label': 'Analysis Failed',
            'risk_level': 'unknown'
        }
        report['indicators'] = []
    
    return report


def calculate_confidence(
    video_confidence: Optional[float] = None,
    audio_confidence: Optional[float] = None,
    face_confidence: Optional[float] = None
) -> float:
    """
    Calculate combined confidence score from multiple analyses.
    
    Args:
        video_confidence: Confidence from video analysis (0-100)
        audio_confidence: Confidence from audio analysis (0-100)
        face_confidence: Confidence from face analysis (0-100)
    
    Returns:
        Combined confidence score (0-100)
    """
    confidences = []
    weights = []
    
    if video_confidence is not None:
        confidences.append(video_confidence)
        weights.append(0.35)
    
    if audio_confidence is not None:
        confidences.append(audio_confidence)
        weights.append(0.30)
    
    if face_confidence is not None:
        confidences.append(face_confidence)
        weights.append(0.35)
    
    if not confidences:
        return 50.0
    
    # Normalize weights
    total_weight = sum(weights[:len(confidences)])
    normalized_weights = [w / total_weight for w in weights[:len(confidences)]]
    
    # Weighted average
    combined = sum(c * w for c, w in zip(confidences, normalized_weights))
    
    return round(combined, 1)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}:{minutes:02d}:00"


def get_risk_emoji(risk_level: str) -> str:
    """Get emoji for risk level"""
    emojis = {
        'low': '✅',
        'medium': '⚠️',
        'high': '🚨',
        'unknown': '❓'
    }
    return emojis.get(risk_level, '❓')


def get_protection_tips(verdict: str) -> List[str]:
    """Get protection tips based on analysis verdict"""
    tips = {
        'likely_fake': [
            "🚨 This content shows signs of manipulation. Do not share without verification.",
            "🔍 Cross-reference with official sources before trusting this media.",
            "📢 Report suspicious content to the platform where you found it.",
            "🛡️ Be skeptical of sensational or emotionally provocative content.",
            "👥 Consult with others before making decisions based on this content."
        ],
        'uncertain': [
            "⚠️ This content has unclear authenticity. Proceed with caution.",
            "🔍 Look for the original source of this media.",
            "📋 Check if multiple reliable sources confirm the content.",
            "🤔 Consider the context - who shared this and why?",
            "📱 Use reverse image/video search to find the original."
        ],
        'likely_authentic': [
            "✅ This content appears genuine, but always stay vigilant.",
            "🔄 Deepfake technology is evolving - stay informed.",
            "🛡️ Enable two-factor authentication on your accounts.",
            "📚 Learn about deepfake detection methods.",
            "👀 Trust but verify - especially for high-stakes content."
        ]
    }
    return tips.get(verdict, tips['uncertain'])


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove path separators and dangerous characters
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    
    sanitized = filename
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Limit length
    if len(sanitized) > 100:
        # Keep extension
        parts = sanitized.rsplit('.', 1)
        if len(parts) == 2:
            name, ext = parts
            sanitized = name[:95] + '.' + ext
        else:
            sanitized = sanitized[:100]
    
    return sanitized


def validate_file_extension(filename: str, allowed_video: bool = True, 
                           allowed_audio: bool = True) -> tuple:
    """
    Validate file extension and return file type.
    
    Returns:
        Tuple of (is_valid, file_type, error_message)
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', 
                       '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}  # Added images
    audio_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac', '.wma'}
    
    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    if allowed_video and ext in video_extensions:
        return (True, 'video', None)
    elif allowed_audio and ext in audio_extensions:
        return (True, 'audio', None)
    else:
        allowed = []
        if allowed_video:
            allowed.extend(video_extensions)
        if allowed_audio:
            allowed.extend(audio_extensions)
        return (False, None, f"Invalid file type. Allowed: {', '.join(sorted(allowed))}")
