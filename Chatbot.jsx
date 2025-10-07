import React, { useState, useRef, useEffect } from 'react';
import '../css/Chatbot.css';
import { FiUser, FiCpu, FiSend } from 'react-icons/fi';

const Chatbot = () => {
  const [messages, setMessages] = useState([
    { text: "Hello! How can I help you today?", sender: "bot" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const sendToAI = async (userText) => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText }),
      });
      const data = await response.json();
      return data.reply;
    } catch {
      return "Error connecting to AI backend.";
    } finally {
      setLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMessage = { text: input, sender: "user" };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    const botReply = await sendToAI(input);
    setMessages(prev => [...prev, { text: botReply, sender: "bot" }]);
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h2>Conversation</h2>
      </div>
      <div className="chatbot-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message-wrapper ${message.sender}`}>
            <div className="message-icon">
              {message.sender === "user" ? <FiUser /> : <FiCpu />}
            </div>
            <div className="message-bubble">{message.text}</div>
          </div>
        ))}
        <div ref={messagesEndRef}></div>
      </div>
      <div className="chatbot-input-area">
        <div className="chatbot-input-wrapper">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyPress={e => e.key === "Enter" && handleSend()}
            placeholder="Type your message here..."
            disabled={loading}
          />
          <button className="send-button" onClick={handleSend} disabled={loading || !input.trim()}>
            <FiSend />
          </button>
        </div>
        {loading && <div className="chatbot-loading">Thinking...</div>}
      </div>
    </div>
  );
};

export default Chatbot;
