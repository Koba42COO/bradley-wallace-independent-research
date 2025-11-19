import React, { useState, useEffect, useRef } from 'react';
import './LLMChat.css';

function LLMChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [quantumMemoryEnabled, setQuantumMemoryEnabled] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:8000/llm/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(m => ({
            role: m.role,
            content: m.content
          })),
          use_quantum_memory: quantumMemoryEnabled,
          max_length: 200,
          temperature: 0.7
        })
      });

      if (!res.ok) {
        throw new Error('Request failed: ' + res.statusText);
      }

      const data = await res.json();
      
      const assistantMessage = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
        context_length: data.context_length,
        unlimited_context: data.unlimited_context
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError('Failed to send message: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="llm-chat-container">
      <div className="chat-header">
        <h3>🧠 PAC LLM Chat</h3>
        <div className="chat-controls">
          <label className="quantum-memory-toggle">
            <input
              type="checkbox"
              checked={quantumMemoryEnabled}
              onChange={(e) => setQuantumMemoryEnabled(e.target.checked)}
            />
            <span>Quantum Memory</span>
          </label>
          <button onClick={clearChat} className="clear-button">Clear</button>
        </div>
      </div>

      <div className="messages-container">
        {messages.length === 0 && (
          <div className="empty-chat">
            <p>Start a conversation with AIVA PAC LLM</p>
            <p className="subtitle">Unlimited context through quantum memory</p>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-content">
              <div className="message-role">
                {msg.role === 'user' ? '👤 You' : '🧠 AIVA'}
              </div>
              <div className="message-text">{msg.content}</div>
              {msg.unlimited_context && (
                <div className="context-badge">
                  ∞ Unlimited Context ({msg.context_length} tokens)
                </div>
              )}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="message assistant">
            <div className="message-content">
              <div className="message-role">🧠 AIVA</div>
              <div className="loading-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      <div className="chat-input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
          rows="2"
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="send-button"
        >
          {loading ? '⏳' : '🚀'}
        </button>
      </div>
    </div>
  );
}

export default LLMChat;

