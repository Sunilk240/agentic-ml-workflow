const LoadingSpinner = () => {
  return (
    <div className="loading-container">
      <div className="loading-spinner">
        <div className="spinner"></div>
        <span>Thinking...</span>
      </div>
    </div>
  );
};

export default LoadingSpinner;