import React from 'react';

const ModelTrainingDisplay = ({ data }) => {
  if (!data || !data.problem_analysis) return null;

  const { 
    problem_analysis, 
    algorithm_recommendations, 
    dataset_readiness, 
    training_preview,
    target_analysis 
  } = data;

  const getReadinessColor = (score) => {
    if (score >= 0.9) return 'excellent';
    if (score >= 0.7) return 'good';
    if (score >= 0.5) return 'fair';
    return 'poor';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
  };

  return (
    <div className="model-training-display">
      {/* Problem Analysis Header */}
      <div className="problem-analysis-header">
        <div className="problem-type-card">
          <h4>🎯 Problem Analysis</h4>
          <div className="problem-details">
            <div className="problem-type">
              <span className="label">Type:</span>
              <span className={`type-badge ${problem_analysis.problem_type}`}>
                {problem_analysis.problem_type.replace('_', ' ').toUpperCase()}
              </span>
            </div>
            <div className="problem-subtype">
              <span className="label">Subtype:</span>
              <span className="subtype-text">
                {problem_analysis.subtype.replace('_', ' ')}
              </span>
            </div>
            <div className="confidence-level">
              <span className="label">Confidence:</span>
              <span className={`confidence-badge ${getConfidenceColor(problem_analysis.confidence)}`}>
                {(problem_analysis.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        <div className="target-analysis-card">
          <h4>🎯 Target Variable</h4>
          <div className="target-details">
            <div className="target-name">
              <span className="label">Column:</span>
              <span className="target-value">{target_analysis.column_name}</span>
            </div>
            <div className="target-stats">
              <div className="stat-item">
                <span className="stat-value">{target_analysis.unique_values}</span>
                <span className="stat-label">Unique Values</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">{target_analysis.data_type}</span>
                <span className="stat-label">Data Type</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Characteristics */}
      <div className="dataset-characteristics">
        <h4>📊 Dataset Characteristics</h4>
        <div className="characteristics-grid">
          <div className="char-item">
            <span className="char-label">Sample Size:</span>
            <span className={`char-value ${problem_analysis.dataset_characteristics.sample_size}`}>
              {problem_analysis.dataset_characteristics.sample_size.toUpperCase()}
            </span>
          </div>
          <div className="char-item">
            <span className="char-label">Features:</span>
            <span className="char-value">{problem_analysis.dataset_characteristics.feature_count}</span>
          </div>
          <div className="char-item">
            <span className="char-label">Complexity:</span>
            <span className={`char-value ${problem_analysis.dataset_characteristics.data_complexity}`}>
              {problem_analysis.dataset_characteristics.data_complexity.replace('_', ' ')}
            </span>
          </div>
          <div className="char-item">
            <span className="char-label">Missing Values:</span>
            <span className={`char-value ${problem_analysis.dataset_characteristics.missing_values > 0 ? 'warning' : 'good'}`}>
              {problem_analysis.dataset_characteristics.missing_values}
            </span>
          </div>
        </div>

        {/* Class Balance for Classification */}
        {problem_analysis.dataset_characteristics.class_balance && (
          <div className="class-balance-section">
            <h5>⚖️ Class Balance</h5>
            <div className="balance-info">
              <div className="balance-status">
                <span className={`balance-badge ${problem_analysis.dataset_characteristics.class_balance.is_balanced ? 'balanced' : 'imbalanced'}`}>
                  {problem_analysis.dataset_characteristics.class_balance.is_balanced ? 'Balanced' : 'Imbalanced'}
                </span>
              </div>
              <div className="class-distribution">
                {problem_analysis.dataset_characteristics.class_balance.classes.map((className, index) => (
                  <div key={className} className="class-item">
                    <span className="class-name">{className}</span>
                    <span className="class-count">
                      {problem_analysis.dataset_characteristics.class_balance.counts[index]} 
                      ({problem_analysis.dataset_characteristics.class_balance.percentages[index]}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Algorithm Recommendations */}
      <div className="algorithm-recommendations">
        <h4>🤖 Algorithm Recommendations</h4>
        <div className="recommendations-summary">
          <div className="summary-stats">
            <span className="recommended-count">{algorithm_recommendations.recommended_count} Recommended</span>
            <span className="total-count">{algorithm_recommendations.total_available} Total Available</span>
          </div>
        </div>

        <div className="algorithms-grid">
          {Object.entries(algorithm_recommendations.algorithms).map(([algName, details]) => (
            <div key={algName} className={`algorithm-card ${details.is_recommended ? 'recommended' : 'alternative'}`}>
              <div className="algorithm-header">
                <h5>{details.name}</h5>
                {details.is_recommended && <span className="recommended-badge">⭐ RECOMMENDED</span>}
                <span className={`confidence-score ${getConfidenceColor(details.confidence)}`}>
                  {(details.confidence * 100).toFixed(0)}%
                </span>
              </div>
              
              <div className="algorithm-characteristics">
                <div className="char-row">
                  <span className="char-label">Training Time:</span>
                  <span className={`char-badge ${details.training_time}`}>{details.training_time}</span>
                </div>
                <div className="char-row">
                  <span className="char-label">Interpretability:</span>
                  <span className={`char-badge ${details.interpretability}`}>{details.interpretability}</span>
                </div>
              </div>

              <div className="pros-cons">
                <div className="pros">
                  <h6>✅ Pros</h6>
                  <ul>
                    {details.pros.map((pro, index) => (
                      <li key={index}>{pro}</li>
                    ))}
                  </ul>
                </div>
                <div className="cons">
                  <h6>⚠️ Cons</h6>
                  <ul>
                    {details.cons.map((con, index) => (
                      <li key={index}>{con}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="suitability">
                <h6>🎯 Best For</h6>
                <div className="suitability-tags">
                  {details.suitable_for.small_data && <span className="suit-tag">Small Data</span>}
                  {details.suitable_for.large_data && <span className="suit-tag">Large Data</span>}
                  {details.suitable_for.interpretability && <span className="suit-tag">Interpretability</span>}
                  {details.suitable_for.high_accuracy && <span className="suit-tag">High Accuracy</span>}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset Readiness */}
      <div className="dataset-readiness">
        <h4>✅ Dataset Readiness</h4>
        <div className="readiness-grid">
          <div className="readiness-item">
            <span className="readiness-label">Features Ready:</span>
            <span className={`readiness-status ${dataset_readiness.features_ready ? 'ready' : 'not-ready'}`}>
              {dataset_readiness.features_ready ? '✅' : '❌'}
            </span>
          </div>
          <div className="readiness-item">
            <span className="readiness-label">Target Identified:</span>
            <span className={`readiness-status ${dataset_readiness.target_identified ? 'ready' : 'not-ready'}`}>
              {dataset_readiness.target_identified ? '✅' : '❌'}
            </span>
          </div>
          <div className="readiness-item">
            <span className="readiness-label">Preprocessing:</span>
            <span className={`readiness-status ${dataset_readiness.preprocessing_complete ? 'ready' : 'not-ready'}`}>
              {dataset_readiness.preprocessing_complete ? '✅ Complete' : '⚠️ Recommended'}
            </span>
          </div>
          <div className="readiness-item">
            <span className="readiness-label">Data Quality:</span>
            <span className={`readiness-score ${getReadinessColor(dataset_readiness.data_quality_score)}`}>
              {(dataset_readiness.data_quality_score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Training Preview */}
      <div className="training-preview">
        <h4>⏱️ Training Preview</h4>
        <div className="preview-grid">
          <div className="preview-item">
            <span className="preview-label">Estimated Time:</span>
            <span className="preview-value">{training_preview.estimated_time}</span>
          </div>
          <div className="preview-item">
            <span className="preview-label">Memory Usage:</span>
            <span className="preview-value">{training_preview.memory_requirements}</span>
          </div>
          <div className="preview-item">
            <span className="preview-label">Training Samples:</span>
            <span className="preview-value">{training_preview.training_samples.toLocaleString()}</span>
          </div>
          <div className="preview-item">
            <span className="preview-label">Test Samples:</span>
            <span className="preview-value">{training_preview.test_samples.toLocaleString()}</span>
          </div>
          <div className="preview-item">
            <span className="preview-label">Cross Validation:</span>
            <span className="preview-value">{training_preview.cross_validation_folds} folds</span>
          </div>
        </div>
      </div>

      {/* Sample Values Preview */}
      {target_analysis.sample_values && target_analysis.sample_values.length > 0 && (
        <div className="sample-values">
          <h4>👀 Target Sample Values</h4>
          <div className="sample-tags">
            {target_analysis.sample_values.slice(0, 8).map((value, index) => (
              <span key={index} className="sample-tag">
                {String(value)}
              </span>
            ))}
            {target_analysis.sample_values.length > 8 && (
              <span className="more-samples">+{target_analysis.sample_values.length - 8} more</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelTrainingDisplay;