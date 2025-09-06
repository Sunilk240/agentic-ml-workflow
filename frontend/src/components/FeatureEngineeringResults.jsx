import React from 'react';

const FeatureEngineeringResults = ({ data }) => {
  if (!data || !data.transformation_summary) return null;

  const { 
    transformation_summary, 
    before_after_comparison, 
    feature_statistics,
    transformation_log,
    correlation_improvement,
    quality_metrics
  } = data;

  return (
    <div className="feature-engineering-results">
      {/* Transformation Summary */}
      <div className="results-header">
        <h4>✅ Feature Engineering Complete</h4>
        <div className="shape-change">
          <span className="shape-before">
            {before_after_comparison.shape_before[0]} × {before_after_comparison.shape_before[1]}
          </span>
          <span className="arrow">→</span>
          <span className="shape-after">
            {before_after_comparison.shape_after[0]} × {before_after_comparison.shape_after[1]}
          </span>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-value">{transformation_summary.features_after}</span>
          <span className="metric-label">Final Features</span>
          <span className={`metric-change ${quality_metrics.feature_count_change >= 0 ? 'positive' : 'negative'}`}>
            {quality_metrics.feature_count_change >= 0 ? '+' : ''}{quality_metrics.feature_count_change}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-value">{transformation_summary.operations_count}</span>
          <span className="metric-label">Operations</span>
        </div>

        <div className="metric-card">
          <span className="metric-value">{quality_metrics.data_retention_rate.toFixed(1)}%</span>
          <span className="metric-label">Data Retained</span>
        </div>

        {correlation_improvement.improvement !== undefined && (
          <div className="metric-card">
            <span className="metric-value">{correlation_improvement.improvement}</span>
            <span className="metric-label">Correlations Reduced</span>
          </div>
        )}
      </div>

      {/* Transformation Log */}
      {transformation_log && transformation_log.length > 0 && (
        <div className="results-section">
          <h5>🔄 Transformations Applied</h5>
          <div className="transformation-timeline">
            {transformation_log.map((log, index) => (
              <div key={index} className="transformation-step">
                <div className="step-icon">
                  {log.operation === 'Encoding' && '🏷️'}
                  {log.operation === 'Scaling' && '📏'}
                  {log.operation === 'Correlation' && '🔗'}
                </div>
                <div className="step-content">
                  <h6>{log.operation}</h6>
                  <p>{log.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Before/After Comparison */}
      <div className="results-section">
        <h5>📊 Before vs After Comparison</h5>
        <div className="comparison-grid">
          <div className="comparison-card">
            <h6>Numeric Features</h6>
            <div className="feature-comparison">
              <div className="before-after">
                <span className="label">Before:</span>
                <span className="count">{before_after_comparison.numeric_features_before.count}</span>
              </div>
              <div className="before-after">
                <span className="label">After:</span>
                <span className="count">{before_after_comparison.numeric_features_after.count}</span>
              </div>
              <div className={`change ${quality_metrics.numeric_feature_change >= 0 ? 'positive' : 'negative'}`}>
                {quality_metrics.numeric_feature_change >= 0 ? '+' : ''}{quality_metrics.numeric_feature_change}
              </div>
            </div>
          </div>

          <div className="comparison-card">
            <h6>Categorical Features</h6>
            <div className="feature-comparison">
              <div className="before-after">
                <span className="label">Before:</span>
                <span className="count">{before_after_comparison.categorical_features_before.count}</span>
              </div>
              <div className="before-after">
                <span className="label">After:</span>
                <span className="count">{before_after_comparison.categorical_features_after.count}</span>
              </div>
              <div className={`change ${quality_metrics.categorical_feature_change >= 0 ? 'positive' : 'negative'}`}>
                {quality_metrics.categorical_feature_change >= 0 ? '+' : ''}{quality_metrics.categorical_feature_change}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Changes */}
      {(transformation_summary.features_added.length > 0 || transformation_summary.features_removed.length > 0) && (
        <div className="results-section">
          <h5>🔄 Feature Changes</h5>
          <div className="feature-changes">
            {transformation_summary.features_added.length > 0 && (
              <div className="change-group">
                <h6>➕ Features Added ({transformation_summary.features_added.length})</h6>
                <div className="feature-list">
                  {transformation_summary.features_added.slice(0, 10).map(feature => (
                    <span key={feature} className="feature-tag added">{feature}</span>
                  ))}
                  {transformation_summary.features_added.length > 10 && (
                    <span className="more-indicator">+{transformation_summary.features_added.length - 10} more</span>
                  )}
                </div>
              </div>
            )}

            {transformation_summary.features_removed.length > 0 && (
              <div className="change-group">
                <h6>➖ Features Removed ({transformation_summary.features_removed.length})</h6>
                <div className="feature-list">
                  {transformation_summary.features_removed.slice(0, 10).map(feature => (
                    <span key={feature} className="feature-tag removed">{feature}</span>
                  ))}
                  {transformation_summary.features_removed.length > 10 && (
                    <span className="more-indicator">+{transformation_summary.features_removed.length - 10} more</span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Correlation Improvement */}
      {correlation_improvement && !correlation_improvement.error && (
        <div className="results-section">
          <h5>🔗 Correlation Analysis</h5>
          <div className="correlation-improvement">
            <div className="correlation-stats">
              <div className="stat-item">
                <span className="stat-label">High Correlations Before:</span>
                <span className="stat-value">{correlation_improvement.high_correlations_before}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">High Correlations After:</span>
                <span className="stat-value">{correlation_improvement.high_correlations_after}</span>
              </div>
              <div className="stat-item improvement">
                <span className="stat-label">Improvement:</span>
                <span className={`stat-value ${correlation_improvement.improvement > 0 ? 'positive' : 'neutral'}`}>
                  {correlation_improvement.improvement > 0 ? '-' : ''}{Math.abs(correlation_improvement.improvement)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Engineered Data Preview */}
      {data.engineered_data_preview && data.engineered_data_preview.length > 0 && (
        <div className="results-section">
          <h5>👀 Engineered Data Preview</h5>
          <div className="data-preview-table">
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    {Object.keys(data.engineered_data_preview[0]).map(col => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.engineered_data_preview.slice(0, 3).map((row, index) => (
                    <tr key={index}>
                      {Object.values(row).map((value, colIndex) => (
                        <td key={colIndex}>
                          {typeof value === 'number' ? value.toFixed(3) : String(value)}
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

      {/* Statistics Comparison */}
      {feature_statistics.before_stats && feature_statistics.after_stats && 
       Object.keys(feature_statistics.before_stats).length > 0 && (
        <div className="results-section">
          <h5>📈 Statistical Changes</h5>
          <div className="stats-comparison">
            <div className="stats-note">
              <p>Feature statistics have been updated after engineering transformations. 
                 Scaled features now have normalized distributions.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FeatureEngineeringResults;