import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [message, setMessage] = useState('');

  // Fetch the welcome message from the Flask backend
  React.useEffect(() => {
    axios.get('/api/hello')
      .then(response => {
        setMessage(response.data.message);
      })
      .catch(error => {
        console.error('There was an error fetching the message!', error);
      });
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handlePredict = async () => {
    if (!file) {
      alert('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('There was an error uploading the file!', error);
    }
  };

  return (
    <div className="App">
      <header>
        <h1>{message}</h1>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button onClick={handlePredict}>Predict</button>
        {prediction && <p>Prediction: {prediction}</p>}
      </header>
    </div>
  );
}

export default App;
