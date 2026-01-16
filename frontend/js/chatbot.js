/**
 * DeepGuard - AI Chatbot Module
 * Educational chatbot for deepfake awareness and guidance
 */

// Chatbot Configuration
const CHATBOT_CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    MAX_MESSAGE_LENGTH: 500,
    TYPING_DELAY: 50,
    MESSAGE_DELAY: 300
};

// Chatbot State
let chatbotState = {
    isOpen: false,
    isTyping: false,
    messageHistory: [],
    initialized: false
};

// DOM Elements
let chatbotElements = {};

/**
 * Initialize chatbot when DOM is ready
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeChatbot();
});

/**
 * Initialize chatbot elements and event listeners
 */
function initializeChatbot() {
    // Get DOM elements
    chatbotElements = {
        widget: document.getElementById('chatbotWidget'),
        toggle: document.getElementById('chatbotToggle'),
        window: document.getElementById('chatbotWindow'),
        close: document.getElementById('chatbotClose'),
        messages: document.getElementById('chatbotMessages'),
        suggestions: document.getElementById('chatbotSuggestions'),
        input: document.getElementById('chatbotInput'),
        send: document.getElementById('chatbotSend')
    };

    // Check if elements exist
    if (!chatbotElements.toggle) {
        console.warn('Chatbot elements not found');
        return;
    }

    // Add event listeners
    chatbotElements.toggle.addEventListener('click', toggleChatbot);
    chatbotElements.close.addEventListener('click', closeChatbot);
    chatbotElements.send.addEventListener('click', sendMessage);
    chatbotElements.input.addEventListener('keypress', handleInputKeypress);
    chatbotElements.input.addEventListener('input', handleInputChange);

    chatbotState.initialized = true;
    console.log('🤖 GuardBot initialized');
}

/**
 * Toggle chatbot window
 */
function toggleChatbot() {
    if (chatbotState.isOpen) {
        closeChatbot();
    } else {
        openChatbot();
    }
}

/**
 * Open chatbot window
 */
function openChatbot() {
    chatbotState.isOpen = true;
    chatbotElements.window.classList.remove('hidden');
    chatbotElements.toggle.classList.add('active');
    chatbotElements.input.focus();

    // Show welcome message if first time
    if (chatbotState.messageHistory.length === 0) {
        showWelcomeMessage();
    }
}

/**
 * Close chatbot window
 */
function closeChatbot() {
    chatbotState.isOpen = false;
    chatbotElements.window.classList.add('hidden');
    chatbotElements.toggle.classList.remove('active');
}

/**
 * Show welcome message
 */
async function showWelcomeMessage() {
    try {
        const response = await fetch(`${CHATBOT_CONFIG.API_BASE_URL}/api/chat/welcome`);
        const data = await response.json();

        if (data.success) {
            addBotMessage(data.message);
            showSuggestions(data.suggestions);
        } else {
            addBotMessage("👋 Hi! I'm **GuardBot**, your AI assistant for deepfake education. How can I help you today?");
            showDefaultSuggestions();
        }
    } catch (error) {
        console.error('Error fetching welcome message:', error);
        addBotMessage("👋 Hi! I'm **GuardBot**, your AI assistant for deepfake education. How can I help you today?");
        showDefaultSuggestions();
    }
}

/**
 * Show default suggestions
 */
function showDefaultSuggestions() {
    showSuggestions([
        "What is a deepfake?",
        "How to detect them?",
        "Protection tips"
    ]);
}

/**
 * Handle input keypress
 */
function handleInputKeypress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

/**
 * Handle input change
 */
function handleInputChange() {
    const hasText = chatbotElements.input.value.trim().length > 0;
    chatbotElements.send.disabled = !hasText;
}

// ==================== OFFLINE KNOWLEDGE BASE ====================
const OFFLINE_RESPONSES = {
    'deepfake': {
        response: "**What is a Deepfake?**\n\nA deepfake is synthetic media created using AI/ML techniques like deep learning. The term combines \"deep learning\" and \"fake.\"\n\n**Common Types:**\n• **Face swaps** - Replacing one person's face with another\n• **Lip sync** - Making someone appear to say different words\n• **Voice cloning** - Synthesizing someone's voice\n• **Full-body puppetry** - Controlling someone's entire appearance",
        suggestions: ["How are deepfakes made?", "How to detect them?", "What are the risks?"]
    },
    'detect': {
        response: "**How to Detect Deepfakes:**\n\n👁️ **Visual Signs:**\n• Unnatural blinking or eye movements\n• Blurry face edges or hairline\n• Inconsistent lighting/shadows\n• Skin texture irregularities\n\n👂 **Audio Signs:**\n• Robotic or unnatural voice patterns\n• Mismatched lip movements\n• Unusual breathing patterns\n\n🔧 **Tools:**\n• Use DeepGuard to analyze suspicious media\n• Reverse image search\n• Check the source credibility",
        suggestions: ["Protection tips", "What is a deepfake?", "How do I report one?"]
    },
    'protect': {
        response: "**Protection Tips:**\n\n🛡️ **Stay Safe:**\n1. **Be skeptical** - Question unusual or shocking content\n2. **Verify sources** - Check multiple reliable sources\n3. **Limit exposure** - Be careful sharing personal photos/videos\n4. **Use detection tools** - Regularly analyze suspicious content\n5. **Stay informed** - Keep up with deepfake technology advances\n\n📱 **If you find a deepfake:**\n• Don't share it further\n• Report to the platform\n• Alert the targeted person\n• Document the source",
        suggestions: ["What is a deepfake?", "How to detect them?", "What are the risks?"]
    },
    'risk': {
        response: "**Risks and Dangers of Deepfakes:**\n\n⚠️ **Personal Harm:**\n• Non-consensual intimate imagery\n• Identity theft and fraud\n• Reputation damage\n• Emotional distress\n\n🏛️ **Societal Impact:**\n• Political manipulation\n• Election interference\n• Fake news spread\n• Erosion of trust in media\n\n💼 **Financial:**\n• CEO fraud scams\n• Stock manipulation\n• Blackmail/extortion",
        suggestions: ["How to protect myself?", "Laws about deepfakes", "How to detect them?"]
    },
    'law': {
        response: "**Deepfake Laws & Regulations:**\n\n🇺🇸 **United States:**\n• DEEPFAKES Accountability Act (proposed)\n• State laws in CA, TX, VA, NY\n• Section 230 debates ongoing\n\n🇪🇺 **European Union:**\n• AI Act includes deepfake provisions\n• GDPR for data protection\n• Digital Services Act\n\n🌍 **Global:**\n• Varies widely by country\n• Many nations developing legislation\n• International cooperation efforts",
        suggestions: ["What are the risks?", "How to report deepfakes?", "Protection tips"]
    },
    'how made': {
        response: "**How Deepfakes Are Made:**\n\n🤖 **Technology:**\n• **GANs** (Generative Adversarial Networks)\n• **Autoencoders** for face swapping\n• **Neural networks** for voice cloning\n\n📊 **Process:**\n1. Collect training data (photos/videos/audio)\n2. Train AI model on the data\n3. Generate synthetic content\n4. Refine and post-process\n\n⏱️ **Requirements:**\n• Lots of source material\n• Powerful computing (GPUs)\n• Technical knowledge\n• Time for training",
        suggestions: ["How to detect them?", "What are the risks?", "Protection tips"]
    },
    'guardbot': {
        response: "**About DeepGuard:**\n\n🛡️ **What We Do:**\nDeepGuard is an AI-powered deepfake detection system that analyzes videos and audio for signs of manipulation.\n\n🔬 **Our Technology:**\n• **CNN Video Analysis** - Detects visual artifacts\n• **Spectral Audio Analysis** - Identifies voice patterns\n• **468-Point Face Tracking** - Analyzes facial movements\n\n🎯 **Our Mission:**\nTo help people identify fake media and protect digital truth.",
        suggestions: ["How to use DeepGuard?", "What is a deepfake?", "Protection tips"]
    },
    'default': {
        response: "I can help you learn about deepfakes! Here are some topics I can assist with:\n\n• **Understanding deepfakes** - What they are and how they work\n• **Detection techniques** - How to spot fake media\n• **Protection tips** - How to stay safe\n• **Laws & regulations** - Legal landscape\n• **Using DeepGuard** - How our tool works\n\nWhat would you like to know more about?",
        suggestions: ["What is a deepfake?", "How to detect them?", "Protection tips"]
    }
};

/**
 * Get offline response based on user message
 */
function getOfflineResponse(message) {
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('deepfake') || lowerMessage.includes('what is') || lowerMessage.includes('what are')) {
        return OFFLINE_RESPONSES['deepfake'];
    }
    if (lowerMessage.includes('detect') || lowerMessage.includes('spot') || lowerMessage.includes('identify') || lowerMessage.includes('tell')) {
        return OFFLINE_RESPONSES['detect'];
    }
    if (lowerMessage.includes('protect') || lowerMessage.includes('safe') || lowerMessage.includes('tip')) {
        return OFFLINE_RESPONSES['protect'];
    }
    if (lowerMessage.includes('risk') || lowerMessage.includes('danger') || lowerMessage.includes('harm')) {
        return OFFLINE_RESPONSES['risk'];
    }
    if (lowerMessage.includes('law') || lowerMessage.includes('legal') || lowerMessage.includes('regulation')) {
        return OFFLINE_RESPONSES['law'];
    }
    if (lowerMessage.includes('made') || lowerMessage.includes('create') || lowerMessage.includes('how are')) {
        return OFFLINE_RESPONSES['how made'];
    }
    if (lowerMessage.includes('deepguard') || lowerMessage.includes('guardbot') || lowerMessage.includes('this tool')) {
        return OFFLINE_RESPONSES['guardbot'];
    }
    if (lowerMessage.includes('hello') || lowerMessage.includes('hi') || lowerMessage.includes('hey')) {
        return {
            response: "Hello! 👋 I'm GuardBot, your AI assistant for deepfake education. How can I help you today?",
            suggestions: ["What is a deepfake?", "How to detect them?", "Protection tips"]
        };
    }
    if (lowerMessage.includes('thank')) {
        return {
            response: "You're welcome! 😊 Stay vigilant and remember - critical thinking is your best defense against deepfakes. Is there anything else you'd like to know?",
            suggestions: ["What is a deepfake?", "Protection tips", "Laws about deepfakes"]
        };
    }

    return OFFLINE_RESPONSES['default'];
}

/**
 * Send message to chatbot
 */
async function sendMessage() {
    const message = chatbotElements.input.value.trim();

    if (!message || chatbotState.isTyping) return;

    // Clear input
    chatbotElements.input.value = '';
    chatbotElements.send.disabled = true;

    // Add user message
    addUserMessage(message);

    // Hide suggestions
    clearSuggestions();

    // Show typing indicator
    showTypingIndicator();

    try {
        const response = await fetch(`${CHATBOT_CONFIG.API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        // Remove typing indicator
        hideTypingIndicator();

        if (data.success) {
            // Show response from server
            await typeMessage(data.response);

            // Show suggestions
            if (data.suggestions && data.suggestions.length > 0) {
                showSuggestions(data.suggestions);
            }
        } else {
            // Fallback to offline mode
            const offlineData = getOfflineResponse(message);
            await typeMessage(offlineData.response);
            showSuggestions(offlineData.suggestions);
        }
    } catch (error) {
        console.warn('Server unavailable, using offline mode:', error.message);
        hideTypingIndicator();

        // Use offline fallback
        const offlineData = getOfflineResponse(message);
        await typeMessage(offlineData.response);
        showSuggestions(offlineData.suggestions);
    }
}

/**
 * Add user message to chat
 */
function addUserMessage(message) {
    const messageEl = document.createElement('div');
    messageEl.className = 'chat-message user-message';
    messageEl.innerHTML = `
        <div class="message-content">${escapeHtml(message)}</div>
        <div class="message-time">${getCurrentTime()}</div>
    `;

    chatbotElements.messages.appendChild(messageEl);
    scrollToBottom();

    chatbotState.messageHistory.push({ role: 'user', content: message });
}

/**
 * Add bot message to chat
 */
function addBotMessage(message) {
    const messageEl = document.createElement('div');
    messageEl.className = 'chat-message bot-message';
    messageEl.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-bubble">
            <div class="message-content">${formatMarkdown(message)}</div>
            <div class="message-time">${getCurrentTime()}</div>
        </div>
    `;

    chatbotElements.messages.appendChild(messageEl);
    scrollToBottom();

    chatbotState.messageHistory.push({ role: 'bot', content: message });
}

/**
 * Type message with animation
 */
async function typeMessage(message) {
    const messageEl = document.createElement('div');
    messageEl.className = 'chat-message bot-message';
    messageEl.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-bubble">
            <div class="message-content"></div>
            <div class="message-time">${getCurrentTime()}</div>
        </div>
    `;

    chatbotElements.messages.appendChild(messageEl);
    const contentEl = messageEl.querySelector('.message-content');

    // For long messages, just show them directly
    if (message.length > 500) {
        contentEl.innerHTML = formatMarkdown(message);
        scrollToBottom();
    } else {
        // Animate shorter messages
        const formattedMessage = formatMarkdown(message);
        contentEl.innerHTML = formattedMessage;
        scrollToBottom();
    }

    chatbotState.messageHistory.push({ role: 'bot', content: message });
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
    chatbotState.isTyping = true;

    const typingEl = document.createElement('div');
    typingEl.className = 'chat-message bot-message typing-indicator';
    typingEl.id = 'typingIndicator';
    typingEl.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-bubble">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;

    chatbotElements.messages.appendChild(typingEl);
    scrollToBottom();
}

/**
 * Hide typing indicator
 */
function hideTypingIndicator() {
    chatbotState.isTyping = false;
    const typingEl = document.getElementById('typingIndicator');
    if (typingEl) {
        typingEl.remove();
    }
}

/**
 * Show suggestions
 */
function showSuggestions(suggestions) {
    chatbotElements.suggestions.innerHTML = '';

    suggestions.forEach(suggestion => {
        const btn = document.createElement('button');
        btn.className = 'suggestion-btn';
        btn.textContent = suggestion;
        btn.addEventListener('click', () => {
            chatbotElements.input.value = suggestion;
            sendMessage();
        });
        chatbotElements.suggestions.appendChild(btn);
    });
}

/**
 * Clear suggestions
 */
function clearSuggestions() {
    chatbotElements.suggestions.innerHTML = '';
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    chatbotElements.messages.scrollTop = chatbotElements.messages.scrollHeight;
}

/**
 * Get current time formatted
 */
function getCurrentTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Format markdown-like syntax to HTML
 */
function formatMarkdown(text) {
    return text
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // Line breaks
        .replace(/\n/g, '<br>')
        // Bullet points
        .replace(/^[•\-]\s/gm, '<span class="bullet">•</span> ')
        // Numbered lists
        .replace(/^(\d+)\.\s/gm, '<span class="number">$1.</span> ')
        // Headers (simple)
        .replace(/^#{1,3}\s+(.+)$/gm, '<strong class="header">$1</strong>');
}

/**
 * Sleep utility
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Export for external use
window.ChatBot = {
    open: openChatbot,
    close: closeChatbot,
    toggle: toggleChatbot,
    sendMessage: (msg) => {
        chatbotElements.input.value = msg;
        sendMessage();
    }
};
