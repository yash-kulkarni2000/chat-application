import React, { useState } from 'react';
import './App.css';

function App () {
  const [message, setMessage] = useState('');
  const [classification, setClassification] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const classifyMessage = async () => {
    if (!message) {
      alert('Please enter a message');
      return;
    }

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
      setClassification(`Classification: ${data.classification}`);
    } catch (error) {
      console.error('Error:', error);
      setClassification('Error classifying the message. Please try again.');
    } finally {
      setIsLoading(false);
    } 
  };

  return (
    <div className='App'>
      <header className='App-header'>
        <h1>Hate Speech Classification</h1>
        <textarea value={message} onChange={(e) => setMessage(e.target.value)} placeholder='Type your message here...'/>
        <button onClick={classifyMessage} disabled={isLoading}>
          {isLoading ? 'Classifying...': 'Classify'}
        </button>
        {classification && <p>{classification}</p>}
      </header>
    </div>
  )
}

export default App;
