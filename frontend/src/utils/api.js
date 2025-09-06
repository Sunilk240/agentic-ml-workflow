import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload-dataset', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const sendMessage = async (message, sessionId) => {
  const response = await api.post('/chat', {
    message,
    session_id: sessionId,
  });

  return response.data;
};

export const getSessionInfo = async (sessionId) => {
  const response = await api.get(`/session/${sessionId}/info`);
  return response.data;
};