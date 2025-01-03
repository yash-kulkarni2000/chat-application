import React, { useState } from 'react';
import './App.css';

function App () {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const sendMessage = async () => {
    if (!message) {
      alert('Please enter a message');
      return;
    }

    setChatHistory((prev) => [...prev, {sender: 'User', text: message}]);

    setIsLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:5000/classify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({message}),
      });

      const data = await response.json();

      const isHateSpeech = data.classification === 'Hate Speech';

      if (isHateSpeech) {
        const confirmSend = window.confirm(
          'Your message contains potentially harmful content. Are you sure you want to send it?'
        );
        if (!confirmSend) {
          setChatHistory((prev) => prev.slice(0, -1));
          setIsLoading(false);
          return;
        }
      }
 
      setChatHistory((prev) => [
        ...prev,
        {sender: 'Bot', text: `Classification: ${data.classification}`, isHateSpeech},
        {sender: 'Bot', text: data.reply},
      ]);
    } catch (error){
      console.error('Error:', error);
      setChatHistory((prev) => [
        ...prev,
        {sender: 'Bot', text: 'Error communicating with the server'},
      ]);
    } finally {
      setIsLoading(false);
      setMessage('');
    }

  };


  return (
    <div className='App'>
      <header className='App-header'>
        <h1>No Hate Mate</h1>
        <div className='chat-window'>
          {chatHistory.map((entry, index) => (
            <div key={index} className={`chat-message ${entry.sender === 'User' ? 'user' : 'bot'} 
            ${entry.isHateSpeech ? 'hate-speech': ''}`}>
            <strong>{entry.sender}:</strong>{entry.text}
            </div>
          ))}
        </div>
        <div className='input-area'>
          <textarea value={message} onChange={(e) => setMessage(e.target.value)} placeholder="Type your message here..."/>
          <button onClick={sendMessage} disabled={isLoading}>
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </header>
    </div>
  );
}

export default App;
