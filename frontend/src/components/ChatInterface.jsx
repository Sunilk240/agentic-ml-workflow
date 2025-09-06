import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { sendMessage } from '../utils/api';
import MessageBubble from './MessageBubble';
import LoadingSpinner from './LoadingSpinner';

const ChatInterface = ({ sessionId, filename }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!sessionId) {
      navigate('/');
      return;
    }

    // Welcome message
    setMessages([{
      id: 1,
      type: 'agent',
      content: `Welcome! I've loaded your dataset "${filename}". You can now ask me to explore data, perform cleaning, train models, or any other ML tasks. What would you like to do first?`,
      timestamp: new Date()
    }]);
  }, [sessionId, filename, navigate]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || loading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);

    try {
      const response = await sendMessage(inputMessage, sessionId);
      
      const agentMessage = {
        id: Date.now() + 1,
        type: 'agent',
        content: response.response,
        timestamp: new Date(),
        detailedData: response.detailed_data  // NEW: Store detailed data
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'agent',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="header-info">
          <h2>ML Agent</h2>
          <span className="dataset-name">Dataset: {filename}</span>
        </div>
        <button 
          onClick={() => navigate('/')} 
          className="new-session-btn"
        >
          New Session
        </button>
      </div>

      <div className="messages-container">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {loading && <LoadingSpinner />}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me to explore data, train models, or any ML task..."
            className="message-input"
            rows="1"
            disabled={loading}
          />
          <button 
            onClick={handleSendMessage}
            disabled={loading || !inputMessage.trim()}
            className="send-btn"
          >
            <span>→</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;