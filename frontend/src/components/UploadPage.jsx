import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadDataset } from '../utils/api';

const UploadPage = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const navigate = useNavigate();

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const response = await uploadDataset(file);
      onUploadSuccess(response.session_id, response.filename);
      navigate('/chat');
    } catch (error) {
      alert('Upload failed: ' + error.message);
    } finally {
      setUploading(false);
    }
  };

  const handleUserGuide = () => {
    navigate('/guide');
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-header">
          <h1>ML Agent Pipeline</h1>
          <p>Upload your dataset to get started with machine learning analysis</p>
        </div>

        <div 
          className={`upload-area ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
        >
          <div className="upload-content">
            <div className="upload-icon">📊</div>
            <h3>Drop your CSV file here</h3>
            <p>or</p>
            <label htmlFor="file-input" className="file-label">
              Browse Files
            </label>
            <input
              id="file-input"
              type="file"
              className="file-input"
              accept=".csv"
              onChange={handleFileSelect}
            />
          </div>
        </div>

        {/* NEW: User Guide Button */}
        <div className="guide-section">
          <button 
            className="user-guide-btn" 
            onClick={handleUserGuide}
            type="button"
          >
            📚 User Guide - How to Use ML Pipeline
          </button>
        </div>

        {file && (
          <div className="file-info">
            <div className="file-details">
              <span className="file-name">{file.name}</span>
              <span className="file-size">
                {(file.size / 1024).toFixed(1)} KB
              </span>
            </div>
            <button 
              className="upload-btn" 
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? 'Uploading...' : 'Start Analysis'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPage;
