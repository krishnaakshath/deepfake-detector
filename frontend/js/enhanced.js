/**
 * DeepGuard - Enhanced Features Module
 * Analysis History, Share Results, Export Report
 */

// ==================== Analysis History ====================

const HISTORY_KEY = 'deepguard_analysis_history';
const MAX_HISTORY_ITEMS = 10;

/**
 * Save analysis result to history
 */
function saveToHistory(result, filename) {
    const history = getHistory();

    const historyItem = {
        id: Date.now(),
        filename: filename,
        score: result.overall_result?.authenticity_score || 0,
        verdict: result.overall_result?.verdict || 'unknown',
        timestamp: new Date().toISOString(),
        details: {
            video: result.video_analysis?.authenticity_score,
            audio: result.audio_analysis?.authenticity_score,
            face: result.face_analysis?.authenticity_score
        }
    };

    // Add to beginning
    history.unshift(historyItem);

    // Keep only last N items
    if (history.length > MAX_HISTORY_ITEMS) {
        history.pop();
    }

    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    updateHistoryUI();
}

/**
 * Get analysis history from localStorage
 */
function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch {
        return [];
    }
}

/**
 * Clear all history
 */
function clearHistory() {
    localStorage.removeItem(HISTORY_KEY);
    updateHistoryUI();
}

/**
 * Update history UI
 */
function updateHistoryUI() {
    const container = document.getElementById('historyContainer');
    const emptyState = document.getElementById('historyEmpty');
    const historySection = document.getElementById('history');

    if (!container) return;

    const history = getHistory();

    if (history.length === 0) {
        if (emptyState) emptyState.style.display = 'block';
        if (historySection) historySection.classList.add('hidden');
        return;
    }

    if (historySection) historySection.classList.remove('hidden');
    if (emptyState) emptyState.style.display = 'none';

    // Clear existing items (keep empty state)
    const existingItems = container.querySelectorAll('.history-item');
    existingItems.forEach(item => item.remove());

    // Add history items
    history.forEach(item => {
        const historyEl = createHistoryElement(item);
        container.insertBefore(historyEl, emptyState);
    });
}

/**
 * Create history item element
 */
function createHistoryElement(item) {
    const el = document.createElement('div');
    el.className = 'history-item';

    const icon = item.filename.match(/\.(mp4|avi|mov|webm)$/i) ? '🎬' : '🎵';
    const scoreClass = item.score >= 70 ? 'authentic' : item.score >= 40 ? 'uncertain' : 'fake';
    const date = new Date(item.timestamp).toLocaleDateString();
    const time = new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    el.innerHTML = `
        <span class="history-icon">${icon}</span>
        <div class="history-info">
            <div class="history-filename">${escapeHtml(item.filename)}</div>
            <div class="history-meta">${date} at ${time}</div>
        </div>
        <div class="history-score ${scoreClass}">${item.score}%</div>
    `;

    return el;
}

// ==================== Share Results ====================

/**
 * Share to Twitter/X
 */
function shareToTwitter(score, verdict) {
    const text = `I just analyzed a file for deepfakes using DeepGuard! 🛡️\n\nAuthenticity Score: ${score}%\nVerdict: ${verdict}\n\nCheck if your media is real: `;
    const url = 'https://github.com/capgarrick/deepfake-detector';
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
    window.open(twitterUrl, '_blank', 'width=550,height=420');
}

/**
 * Share to LinkedIn
 */
function shareToLinkedIn(score, verdict) {
    const url = 'https://github.com/capgarrick/deepfake-detector';
    const linkedInUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`;
    window.open(linkedInUrl, '_blank', 'width=550,height=420');
}

/**
 * Copy results to clipboard
 */
function copyResultsToClipboard(result) {
    const text = `DeepGuard Analysis Results
========================
Authenticity Score: ${result.overall_result?.authenticity_score || 0}%
Verdict: ${result.overall_result?.verdict || 'unknown'}
Confidence: ${result.overall_result?.confidence || 0}%

Video Analysis: ${result.video_analysis?.authenticity_score || 'N/A'}%
Audio Analysis: ${result.audio_analysis?.authenticity_score || 'N/A'}%
Face Analysis: ${result.face_analysis?.authenticity_score || 'N/A'}%

Analyzed with DeepGuard - https://github.com/capgarrick/deepfake-detector`;

    navigator.clipboard.writeText(text).then(() => {
        showToast('Results copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy results', 'error');
    });
}

/**
 * Add share buttons to results
 */
function addShareButtons(result) {
    const resultsSection = document.querySelector('.results-section');
    if (!resultsSection) return;

    // Remove existing share section
    const existingShare = resultsSection.querySelector('.share-section');
    if (existingShare) existingShare.remove();

    const shareSection = document.createElement('div');
    shareSection.className = 'share-section';
    shareSection.innerHTML = `
        <p class="share-title">Share your results</p>
        <div class="share-buttons">
            <button class="share-btn twitter" id="shareTwitter">
                <span>𝕏</span> Share on X
            </button>
            <button class="share-btn linkedin" id="shareLinkedIn">
                <span>in</span> LinkedIn
            </button>
            <button class="share-btn copy" id="copyResults">
                📋 Copy Results
            </button>
        </div>
    `;

    resultsSection.appendChild(shareSection);

    // Add event listeners
    const score = result.overall_result?.authenticity_score || 0;
    const verdict = result.overall_result?.verdict || 'unknown';

    document.getElementById('shareTwitter')?.addEventListener('click', () => shareToTwitter(score, verdict));
    document.getElementById('shareLinkedIn')?.addEventListener('click', () => shareToLinkedIn(score, verdict));
    document.getElementById('copyResults')?.addEventListener('click', () => copyResultsToClipboard(result));
}

// ==================== Export Report ====================

/**
 * Export results as text file
 */
function exportReport(result, filename) {
    const report = `
╔══════════════════════════════════════════════════════════════╗
║                    DEEPGUARD ANALYSIS REPORT                  ║
╚══════════════════════════════════════════════════════════════╝

FILE INFORMATION
────────────────
Filename: ${filename}
Analysis Date: ${new Date().toLocaleString()}

OVERALL RESULTS
───────────────
Authenticity Score: ${result.overall_result?.authenticity_score || 0}%
Verdict: ${(result.overall_result?.verdict || 'UNKNOWN').toUpperCase()}
Confidence: ${result.overall_result?.confidence || 0}%

DETAILED ANALYSIS
─────────────────
${result.video_analysis ? `
VIDEO ANALYSIS
  Score: ${result.video_analysis.authenticity_score}%
  Compression Artifacts: ${result.video_analysis.indicators?.compression_artifacts || 'N/A'}
  Color Consistency: ${result.video_analysis.indicators?.color_consistency || 'N/A'}
  Noise Patterns: ${result.video_analysis.indicators?.noise_patterns || 'N/A'}
` : '  Video analysis not performed.'}
${result.audio_analysis ? `
AUDIO ANALYSIS
  Score: ${result.audio_analysis.authenticity_score}%
  Voice Naturalness: ${result.audio_analysis.indicators?.voice_naturalness || 'N/A'}
  Spectral Quality: ${result.audio_analysis.indicators?.spectral_quality || 'N/A'}
` : '  Audio analysis not performed.'}
${result.face_analysis ? `
FACE LANDMARK ANALYSIS
  Score: ${result.face_analysis.authenticity_score}%
  Blink Pattern: ${result.face_analysis.indicators?.blink_pattern || 'N/A'}
  Lip Sync: ${result.face_analysis.indicators?.lip_sync || 'N/A'}
  Facial Symmetry: ${result.face_analysis.indicators?.facial_symmetry || 'N/A'}
` : '  Face analysis not performed.'}

RECOMMENDATIONS
───────────────
${result.protection_tips?.slice(0, 3).join('\n') || 'No specific recommendations.'}

────────────────────────────────────────────────────────────────
Report generated by DeepGuard - AI Deepfake Detection System
https://github.com/capgarrick/deepfake-detector
────────────────────────────────────────────────────────────────
`;

    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `deepguard-report-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Report downloaded!', 'success');
}

// ==================== Utility Functions ====================

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    // Check if showNotification exists from main app.js
    if (typeof showNotification === 'function') {
        showNotification(message, type);
    } else {
        console.log(`[${type}] ${message}`);
    }
}

// ==================== Initialize ====================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize history UI
    updateHistoryUI();

    // Clear history button
    const clearBtn = document.getElementById('clearHistoryBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear all analysis history?')) {
                clearHistory();
                showToast('History cleared', 'success');
            }
        });
    }

    console.log('📊 Enhanced features module loaded');
});

// Export functions for use in main app
window.DeepGuardEnhanced = {
    saveToHistory,
    getHistory,
    clearHistory,
    addShareButtons,
    exportReport
};
