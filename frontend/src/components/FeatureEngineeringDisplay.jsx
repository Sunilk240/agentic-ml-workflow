import React from 'react';

const FeatureEngineeringDisplay = ({ data }) => {
  if (!data || !data.feature_analysis) return null;

  const { feature_analysis, engineering_opportunities, feature_summary } = data;

  return (
    <div className="feature-engineering-display">
      <div className="feature-summary-cards">
        <div className="summary-card">
          <h4>📊 Feature Overview</h4>
          <div className="stat-grid">
            <div className="stat-item">
              <span className="stat-value">{feature_summary.total_features}</span>
              <span className="stat-label">Total Features</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{feature_summary.numeric_count}</span>
              <span className="stat-label">Numeric</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{feature_summary.categorical_count}</span>
              <span className="stat-label">Categorical</span>
            </div>
          </div>
        </div>

        <div className="summary-card">
          <h4>🔧 Engineering Needed</h4>
          <div className="engineering-status">
            {feature_summary.engineering_needed ? (
              <span className="status-badge needs-engineering">Yes</span>
            ) : (
              <span className="status-badge ready">Ready</span>
            )}
          </div>
        </div>
      </div>

      <div className="feature-analysis-sections">
        {/* Numeric Features Analysis */}
        {feature_analysis.numeric_features.count > 0 && (
          <div className="analysis-section">
            <h4>🔢 Numeric Features Analysis</h4>
            <div className="feature-grid">
              <div className="feature-info">
                <h5>Scaling Requirements</h5>
                <div className="scaling-list">
                  {feature_analysis.numeric_features.scaling_needed.map(feature => (
                    <div key={feature} className="scaling-item">
                      <span className="feature-name">{feature}</span>
                      <span className="scaling-badge">Needs Scaling</span>
                    </div>
                  ))}
                  {feature_analysis.numeric_features.scaling_needed.length === 0 && (
                    <div className="no-scaling">✅ All features properly scaled</div>
                  )}
                </div>
              </div>

              {Object.keys(feature_analysis.numeric_features.scaling_analysis).length > 0 && (
                <div className="feature-info">
                  <h5>Feature Statistics</h5>
                  <div className="stats-table">
                    <div className="stats-header">
                      <span>Feature</span>
                      <span>Range</span>
                      <span>Mean</span>
                      <span>Std</span>
                    </div>
                    {Object.entries(feature_analysis.numeric_features.scaling_analysis)
                      .slice(0, 5)
                      .map(([feature, stats]) => (
                      <div key={feature} className="stats-row">
                        <span className="feature-name">{feature}</span>
                        <span className="stat-value">{stats.range.toFixed(2)}</span>
                        <span className="stat-value">{stats.mean.toFixed(2)}</span>
                        <span className="stat-value">{stats.std.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Categorical Features Analysis */}
        {feature_analysis.categorical_features.count > 0 && (
          <div className="analysis-section">
            <h4>📝 Categorical Features Analysis</h4>
            <div className="categorical-grid">
              <div className="categorical-info">
                <h5>Encoding Candidates</h5>
                <div className="encoding-list">
                  {feature_analysis.categorical_features.names.map(feature => (
                    <div key={feature} className="encoding-item">
                      <span className="feature-name">{feature}</span>
                      <span className="encoding-badge">Needs Encoding</span>
                    </div>
                  ))}
                </div>
              </div>

              {Object.keys(feature_analysis.categorical_features.sample_values).length > 0 && (
                <div className="categorical-info">
                  <h5>Sample Values</h5>
                  <div className="samples-container">
                    {Object.entries(feature_analysis.categorical_features.sample_values).map(([feature, values]) => (
                      <div key={feature} className="sample-group">
                        <h6>{feature}</h6>
                        <div className="sample-values">
                          {Object.entries(values).map(([value, count]) => (
                            <span key={value} className="sample-tag">
                              {value} ({count})
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Feature Relationships */}
        {feature_analysis.feature_relationships.high_correlations.length > 0 && (
          <div className="analysis-section">
            <h4>🔗 Feature Relationships</h4>
            <div className="correlations-info">
              <h5>High Correlations ({feature_analysis.feature_relationships.total_correlations} total)</h5>
              <div className="correlation-list">
                {feature_analysis.feature_relationships.high_correlations.map((corr, index) => (
                  <div key={index} className="correlation-item">
                    <div className="correlation-pair">
                      <span className="feature-name">{corr.feature1}</span>
                      <span className="correlation-arrow">↔</span>
                      <span className="feature-name">{corr.feature2}</span>
                    </div>
                    <span className={`correlation-value ${corr.correlation > 0.9 ? 'very-high' : 'high'}`}>
                      {corr.correlation.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Engineering Opportunities */}
        <div className="analysis-section">
          <h4>💡 Engineering Opportunities</h4>
          <div className="opportunities-grid">
            {engineering_opportunities.scaling_candidates.count > 0 && (
              <div className="opportunity-card">
                <h5>🎯 Scaling</h5>
                <div className="opportunity-stats">
                  <span className="count">{engineering_opportunities.scaling_candidates.count} features</span>
                  <p className="reason">{engineering_opportunities.scaling_candidates.reason}</p>
                </div>
              </div>
            )}

            {engineering_opportunities.encoding_candidates.count > 0 && (
              <div className="opportunity-card">
                <h5>🏷️ Encoding</h5>
                <div className="opportunity-stats">
                  <span className="count">{engineering_opportunities.encoding_candidates.count} features</span>
                  <p className="reason">{engineering_opportunities.encoding_candidates.reason}</p>
                </div>
              </div>
            )}

            {engineering_opportunities.correlation_reduction.count > 0 && (
              <div className="opportunity-card">
                <h5>🔗 Correlation</h5>
                <div className="opportunity-stats">
                  <span className="count">{engineering_opportunities.correlation_reduction.count} pairs</span>
                  <p className="reason">{engineering_opportunities.correlation_reduction.reason}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Data Preview */}
        {data.data_preview && data.data_preview.length > 0 && (
          <div className="analysis-section">
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
    </div>
  );
};

export default FeatureEngineeringDisplay;