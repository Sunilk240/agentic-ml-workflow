import React from 'react';

const ClusteringDisplay = ({ data }) => {
  if (!data || !data.clustering_analysis) return null;

  const { 
    clustering_analysis, 
    algorithm_recommendations, 
    optimal_clusters_analysis,
    feature_analysis 
  } = data;

  const getSuitabilityColor = (score) => {
    if (score >= 0.8) return 'excellent';
    if (score >= 0.6) return 'good';
    if (score >= 0.4) return 'fair';
    return 'poor';
  };

  const getComplexityColor = (complexity) => {
    switch(complexity) {
      case 'low': return 'low';
      case 'medium': return 'medium';
      case 'high': return 'high';
      default: return 'medium';
    }
  };

  const getScalabilityColor = (scalability) => {
    switch(scalability) {
      case 'high': return 'high';
      case 'medium': return 'medium';
      case 'low': return 'low';
      default: return 'medium';
    }
  };

  return (
    <div className="clustering-display">
      {/* Dataset Analysis Header */}
      <div className="clustering-analysis-header">
        <div className="dataset-overview-card">
          <h4>📊 Dataset Overview</h4>
          <div className="dataset-stats">
            <div className="stat-item">
              <span className="stat-value">{clustering_analysis.dataset_characteristics.sample_count.toLocaleString()}</span>
              <span className="stat-label">Samples</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{clustering_analysis.dataset_characteristics.feature_count}</span>
              <span className="stat-label">Features</span>
            </div>
            <div className="stat-item">
              <span className={`stat-value ${clustering_analysis.dataset_characteristics.sample_size}`}>
                {clustering_analysis.dataset_characteristics.sample_size.toUpperCase()}
              </span>
              <span className="stat-label">Size</span>
            </div>
            <div className="stat-item">
              <span className={`stat-value ${clustering_analysis.dataset_characteristics.dimensionality}`}>
                {clustering_analysis.dataset_characteristics.dimensionality.toUpperCase()}
              </span>
              <span className="stat-label">Dimensionality</span>
            </div>
          </div>
        </div>

        <div className="suitability-card">
          <h4>🎯 Clustering Suitability</h4>
          <div className="suitability-score">
            <div className={`score-circle ${getSuitabilityColor(clustering_analysis.clustering_suitability.overall_score)}`}>
              <span className="score-value">
                {(clustering_analysis.clustering_suitability.overall_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="score-interpretation">
              {clustering_analysis.clustering_suitability.interpretation.toUpperCase()}
            </div>
          </div>
        </div>
      </div>

      {/* Suitability Analysis */}
      <div className="suitability-analysis">
        <h4>✅ Suitability Analysis</h4>
        <div className="analysis-grid">
          {clustering_analysis.clustering_suitability.reasons.length > 0 && (
            <div className="analysis-section positive">
              <h5>✅ Strengths</h5>
              <ul>
                {clustering_analysis.clustering_suitability.reasons.map((reason, index) => (
                  <li key={index}>{reason}</li>
                ))}
              </ul>
            </div>
          )}

          {clustering_analysis.clustering_suitability.challenges.length > 0 && (
            <div className="analysis-section challenges">
              <h5>⚠️ Challenges</h5>
              <ul>
                {clustering_analysis.clustering_suitability.challenges.map((challenge, index) => (
                  <li key={index}>{challenge}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="analysis-section preprocessing">
            <h5>🔧 Preprocessing</h5>
            <div className="preprocessing-status">
              {clustering_analysis.clustering_suitability.preprocessing_needed ? (
                <span className="status-badge needed">Required</span>
              ) : (
                <span className="status-badge ready">Ready</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Optimal Clusters Analysis */}
      {optimal_clusters_analysis && (
        <div className="optimal-clusters-section">
          <h4>📈 Optimal Clusters Analysis</h4>
          <div className="clusters-analysis-grid">
            <div className="analysis-method">
              <h5>📊 Elbow Method</h5>
              <div className="method-results">
                <div className="result-item">
                  <span className="result-label">Suggested Range:</span>
                  <span className="result-value">
                    {optimal_clusters_analysis.elbow_method.suggested_range[0]} - {optimal_clusters_analysis.elbow_method.suggested_range[1]} clusters
                  </span>
                </div>
                <div className="result-item">
                  <span className="result-label">Optimal K:</span>
                  <span className="result-value optimal">{optimal_clusters_analysis.elbow_method.optimal_k}</span>
                </div>
              </div>
            </div>

            <div className="analysis-method">
              <h5>🎯 Silhouette Analysis</h5>
              <div className="method-results">
                <div className="result-item">
                  <span className="result-label">Best K:</span>
                  <span className="result-value optimal">{optimal_clusters_analysis.silhouette_analysis.best_k}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Silhouette Score:</span>
                  <span className="result-value">{optimal_clusters_analysis.silhouette_analysis.best_score.toFixed(3)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Cluster Range Visualization */}
          {optimal_clusters_analysis.silhouette_analysis.silhouette_scores && (
            <div className="silhouette-chart">
              <h5>📊 Silhouette Scores by K</h5>
              <div className="chart-container">
                {optimal_clusters_analysis.silhouette_analysis.k_range.map((k, index) => (
                  <div key={k} className="score-bar">
                    <div className="bar-info">
                      <span className="k-value">K={k}</span>
                      <span className="score-value">
                        {optimal_clusters_analysis.silhouette_analysis.silhouette_scores[index]?.toFixed(3) || '0.000'}
                      </span>
                    </div>
                    <div className="bar-container">
                      <div 
                        className={`bar-fill ${k === optimal_clusters_analysis.silhouette_analysis.best_k ? 'optimal' : ''}`}
                        style={{ 
                          width: `${Math.max(0, (optimal_clusters_analysis.silhouette_analysis.silhouette_scores[index] || 0) * 100)}%` 
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Algorithm Recommendations */}
      <div className="algorithm-recommendations">
        <h4>🤖 Algorithm Recommendations</h4>
        
        {/* Recommended Algorithm */}
        <div className="recommended-algorithm">
          <h5>⭐ Recommended: {algorithm_recommendations.recommended.details.name}</h5>
          <div className="algorithm-card recommended">
            <div className="algorithm-header">
              <div className="algorithm-info">
                <span className="confidence-score">
                  {(algorithm_recommendations.recommended.details.confidence * 100).toFixed(0)}% Confidence
                </span>
                <span className="optimal-clusters">
                  Optimal: {algorithm_recommendations.recommended.details.optimal_clusters}
                </span>
              </div>
              <div className="algorithm-metrics">
                <span className={`complexity-badge ${getComplexityColor(algorithm_recommendations.recommended.details.complexity)}`}>
                  {algorithm_recommendations.recommended.details.complexity} complexity
                </span>
                <span className={`scalability-badge ${getScalabilityColor(algorithm_recommendations.recommended.details.scalability)}`}>
                  {algorithm_recommendations.recommended.details.scalability} scalability
                </span>
              </div>
            </div>

            <div className="algorithm-details">
              <div className="reasons">
                <h6>🎯 Why Recommended</h6>
                <ul>
                  {algorithm_recommendations.recommended.details.reasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              </div>

              <div className="pros-cons">
                <div className="pros">
                  <h6>✅ Advantages</h6>
                  <ul>
                    {algorithm_recommendations.recommended.details.pros.map((pro, index) => (
                      <li key={index}>{pro}</li>
                    ))}
                  </ul>
                </div>
                <div className="cons">
                  <h6>⚠️ Limitations</h6>
                  <ul>
                    {algorithm_recommendations.recommended.details.cons.map((con, index) => (
                      <li key={index}>{con}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="best-for">
                <h6>🎯 Best For</h6>
                <div className="best-for-tags">
                  {algorithm_recommendations.recommended.details.best_for.map((use_case, index) => (
                    <span key={index} className="use-case-tag">{use_case}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* All Algorithms Comparison */}
        <div className="all-algorithms">
          <h5>🔄 All Available Algorithms</h5>
          <div className="algorithms-grid">
            {Object.entries(algorithm_recommendations.all_algorithms).map(([algName, details]) => (
              <div key={algName} className={`algorithm-card ${algName === algorithm_recommendations.recommended.name ? 'recommended' : 'alternative'}`}>
                <div className="algorithm-header">
                  <h6>{details.name}</h6>
                  {algName === algorithm_recommendations.recommended.name && (
                    <span className="recommended-badge">⭐ RECOMMENDED</span>
                  )}
                  <span className="confidence-score">
                    {(details.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="algorithm-characteristics">
                  <div className="char-row">
                    <span className="char-label">Complexity:</span>
                    <span className={`char-badge ${getComplexityColor(details.complexity)}`}>
                      {details.complexity}
                    </span>
                  </div>
                  <div className="char-row">
                    <span className="char-label">Scalability:</span>
                    <span className={`char-badge ${getScalabilityColor(details.scalability)}`}>
                      {details.scalability}
                    </span>
                  </div>
                  <div className="char-row">
                    <span className="char-label">Clusters:</span>
                    <span className="char-value">{details.optimal_clusters}</span>
                  </div>
                </div>

                <div className="algorithm-summary">
                  <div className="best-for-compact">
                    <strong>Best for:</strong> {details.best_for[0]}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Feature Analysis */}
      <div className="feature-analysis">
        <h4>🔍 Feature Analysis</h4>
        <div className="feature-info-grid">
          <div className="feature-overview">
            <h5>📊 Feature Overview</h5>
            <div className="feature-stats">
              <div className="stat-item">
                <span className="stat-value">{feature_analysis.feature_count}</span>
                <span className="stat-label">Clustering Features</span>
              </div>
              <div className="stat-item">
                <span className={`stat-value ${feature_analysis.scaling_needed ? 'warning' : 'good'}`}>
                  {feature_analysis.scaling_needed ? 'NEEDED' : 'READY'}
                </span>
                <span className="stat-label">Scaling</span>
              </div>
              <div className="stat-item">
                <span className={`stat-value ${feature_analysis.dimensionality_reduction === 'recommended' ? 'warning' : 'good'}`}>
                  {feature_analysis.dimensionality_reduction.toUpperCase()}
                </span>
                <span className="stat-label">Dim. Reduction</span>
              </div>
            </div>
          </div>

          <div className="feature-list">
            <h5>📋 Features Used</h5>
            <div className="feature-tags">
              {feature_analysis.clustering_features.slice(0, 8).map((feature, index) => (
                <span key={index} className="feature-tag">{feature}</span>
              ))}
              {feature_analysis.clustering_features.length > 8 && (
                <span className="more-features">+{feature_analysis.clustering_features.length - 8} more</span>
              )}
            </div>
          </div>
        </div>

        {/* High Correlations */}
        {feature_analysis.feature_correlations.high_correlations && 
         feature_analysis.feature_correlations.high_correlations.length > 0 && (
          <div className="correlations-section">
            <h5>🔗 High Feature Correlations</h5>
            <div className="correlation-list">
              {feature_analysis.feature_correlations.high_correlations.map((corr, index) => (
                <div key={index} className="correlation-item">
                  <div className="correlation-pair">
                    <span className="feature-name">{corr.feature1}</span>
                    <span className="correlation-arrow">↔</span>
                    <span className="feature-name">{corr.feature2}</span>
                  </div>
                  <span className={`correlation-value ${Math.abs(corr.correlation) > 0.9 ? 'very-high' : 'high'}`}>
                    {corr.correlation.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Data Preview */}
      {data.data_preview && data.data_preview.length > 0 && (
        <div className="data-preview-section">
          <h4>👀 Data Preview</h4>
          <div className="data-preview-table">
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    {Object.keys(data.data_preview[0]).map(col => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.data_preview.slice(0, 3).map((row, index) => (
                    <tr key={index}>
                      {Object.values(row).map((value, colIndex) => (
                        <td key={colIndex}>
                          {typeof value === 'number' ? value.toFixed(2) : String(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ClusteringDisplay;