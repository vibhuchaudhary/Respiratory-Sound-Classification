import React, { useState } from 'react';
import '../css/Chatbot.css';
import { FiUser, FiCpu, FiSend } from 'react-icons/fi';

const Chatbot = () => {
    const [messages, setMessages] = useState([
        { text: "Hello! How can I help you today?", sender: 'bot' }
    ]);
    const [input, setInput] = useState('');

    const handleSend = () => {
        if (input.trim()) {
            const userMessage = { text: input, sender: 'user' };
            setMessages([...messages, userMessage]);
            setInput('');
            
            setTimeout(() => {
                const botMessage = { text: 'This is a simulated response.', sender: 'bot' };
                setMessages(prev => [...prev, botMessage]);
            }, 1000);
        }
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
                            {message.sender === 'user' ? <FiUser /> : <FiCpu />}
                        </div>
                        <div className="message-bubble">
                            {message.text}
                        </div>
                    </div>
                ))}
            </div>
            <div className="chatbot-input-area">
                <div className="chatbot-input-wrapper">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Type your message here..."
                    />
                    <button className="send-button" onClick={handleSend}>
                        <FiSend />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Chatbot;