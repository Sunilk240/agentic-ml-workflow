import FeatureEngineeringDisplay from './FeatureEngineeringDisplay';
import FeatureEngineeringResults from './FeatureEngineeringResults';
import ModelTrainingDisplay from './ModelTrainingDisplay';
import ModelTrainingResults from './ModelTrainingResults';
import ClusteringDisplay from './ClusteringDisplay';
import ClusteringResults from './ClusteringResults';

const MessageBubble = ({ message }) => {
  const formatContent = (content) => {
    // Simple formatting for better readability
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br/>');
  };

  const renderDetailedData = (detailedData) => {
    if (!detailedData) return null;

    return (
      <div className="detailed-data-section">
        <div className="detailed-header">📊 Additional Insights</div>
        
        {/* Data Preview Table */}
        {detailedData.data_preview && (
          <div className="data-preview">
            <h4>📋 Data Preview (First 5 Rows)</h4>
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    {Object.keys(detailedData.data_preview[0] || {}).map(col => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {detailedData.data_preview.map((row, idx) => (
                    <tr key={idx}>
                      {Object.values(row).map((value, colIdx) => (
                        <td key={colIdx}>{String(value)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Column Information */}
        {detailedData.columns_info && (
          <div className="columns-info">
            <h4>📊 Column Details</h4>
            <div className="column-stats">
              <div className="stat-item">
                <span className="stat-label">Numeric Columns:</span>
                <span className="stat-value">{detailedData.columns_info.numeric.count}</span>
                <div className="stat-details">{detailedData.columns_info.numeric.names.join(', ')}</div>
              </div>
              <div className="stat-item">
                <span className="stat-label">Categorical Columns:</span>
                <span className="stat-value">{detailedData.columns_info.categorical.count}</span>
                <div className="stat-details">{detailedData.columns_info.categorical.names.join(', ')}</div>
              </div>
            </div>
          </div>
        )}

        {/* Data Quality */}
        {detailedData.data_quality && (
          <div className="data-quality">
            <h4>🔍 Data Quality Summary</h4>
            <div className="quality-stats">
              <div className="quality-item">
                <span className="quality-label">Missing Values:</span>
                <span className="quality-value">
                  {detailedData.data_quality.missing_values.total} ({detailedData.data_quality.missing_values.percentage}%)
                </span>
              </div>
              <div className="quality-item">
                <span className="quality-label">Duplicate Rows:</span>
                <span className="quality-value">
                  {detailedData.data_quality.duplicates.count} ({detailedData.data_quality.duplicates.percentage}%)
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Cleaning Analysis (for perform_data_cleaning) */}
        {detailedData.issues_analysis && (
          <div className="cleaning-analysis">
            <h4>🔍 Detailed Issue Analysis</h4>
            
            {detailedData.issues_analysis.missing_values && (
              <div className="issue-card">
                <div className="issue-header">
                  <span className="issue-title">Missing Values</span>
                  <span className={`severity-badge ${detailedData.issues_analysis.missing_values.severity.toLowerCase()}`}>
                    {detailedData.issues_analysis.missing_values.severity}
                  </span>
                </div>
                <div className="issue-details">
                  <p><strong>Affected Columns:</strong> {detailedData.issues_analysis.missing_values.affected_columns.join(', ')}</p>
                  <p><strong>Pattern:</strong> {detailedData.issues_analysis.missing_values.pattern}</p>
                </div>
              </div>
            )}

            {detailedData.issues_analysis.duplicates && (
              <div className="issue-card">
                <div className="issue-header">
                  <span className="issue-title">Duplicate Rows</span>
                  <span className={`severity-badge ${detailedData.issues_analysis.duplicates.severity.toLowerCase()}`}>
                    {detailedData.issues_analysis.duplicates.severity}
                  </span>
                </div>
                <div className="issue-details">
                  <p><strong>Count:</strong> {detailedData.issues_analysis.duplicates.count} rows ({detailedData.issues_analysis.duplicates.percentage}%)</p>
                  {detailedData.issues_analysis.duplicates.sample_duplicates.length > 0 && (
                    <div className="sample-data">
                      <p><strong>Sample Duplicates:</strong></p>
                      <div className="mini-table">
                        {detailedData.issues_analysis.duplicates.sample_duplicates.slice(0, 2).map((row, idx) => (
                          <div key={idx} className="mini-row">
                            {Object.entries(row).slice(0, 3).map(([key, value]) => (
                              <span key={key}>{key}: {String(value)}</span>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {detailedData.issues_analysis.outliers && (
              <div className="issue-card">
                <div className="issue-header">
                  <span className="issue-title">Outliers</span>
                  <span className={`severity-badge ${detailedData.issues_analysis.outliers.severity.toLowerCase()}`}>
                    {detailedData.issues_analysis.outliers.severity}
                  </span>
                </div>
                <div className="issue-details">
                  <p><strong>Total Outliers:</strong> {detailedData.issues_analysis.outliers.total_outliers}</p>
                  <p><strong>Affected Columns:</strong> {detailedData.issues_analysis.outliers.affected_columns}</p>
                  <div className="outlier-columns">
                    {Object.entries(detailedData.issues_analysis.outliers.column_wise_outliers).slice(0, 3).map(([col, info]) => (
                      <div key={col} className="outlier-column">
                        <strong>{col}:</strong> {info.count} outliers ({info.percentage}%)
                        {detailedData.issues_analysis.outliers.outlier_samples[col] && (
                          <div className="outlier-samples">
                            Sample values: {detailedData.issues_analysis.outliers.outlier_samples[col].slice(0, 3).join(', ')}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Feature Engineering Analysis (for perform_feature_engineering) */}
        {detailedData.feature_analysis && (
          <FeatureEngineeringDisplay data={detailedData} />
        )}

        {/* Feature Engineering Results (for execute_feature_engineering) */}
        {detailedData.transformation_summary && (
          <FeatureEngineeringResults data={detailedData} />
        )}

        {/* Model Training Analysis (for train_model) */}
        {detailedData.problem_analysis && (
          <ModelTrainingDisplay data={detailedData} />
        )}

        {/* Model Training Results (for execute_model_training) */}
        {detailedData.training_summary && (
          <ModelTrainingResults data={detailedData} />
        )}

        {/* Clustering Analysis (for perform_clustering) */}
        {detailedData.clustering_analysis && (
          <ClusteringDisplay data={detailedData} />
        )}

        {/* Clustering Results (for execute_clustering) */}
        {detailedData.clustering_results && (
          <ClusteringResults data={detailedData} />
        )}

        {/* Cleaning Results (for execute_data_cleaning) */}
        {detailedData.cleaning_summary && (
          <div className="cleaning-results">
            <h4>✅ Cleaning Results</h4>
            
            <div className="summary-stats">
              <div className="stat-card">
                <span className="stat-number">{detailedData.cleaning_summary.rows_before}</span>
                <span className="stat-label">Rows Before</span>
              </div>
              <div className="stat-arrow">→</div>
              <div className="stat-card">
                <span className="stat-number">{detailedData.cleaning_summary.rows_after}</span>
                <span className="stat-label">Rows After</span>
              </div>
              <div className="stat-card improvement">
                <span className="stat-number">{detailedData.quality_improvement.data_retention_rate}%</span>
                <span className="stat-label">Data Retained</span>
              </div>
            </div>

            {detailedData.transformation_log && (
              <div className="transformation-log">
                <h5>🔄 Operations Performed</h5>
                {detailedData.transformation_log.map((op, idx) => (
                  <div key={idx} className="operation-item">
                    <span className="operation-name">{op.operation}</span>
                    <span className="operation-desc">{op.description}</span>
                  </div>
                ))}
              </div>
            )}

            {detailedData.cleaned_data_preview && (
              <div className="cleaned-preview">
                <h5>📋 Cleaned Data Preview</h5>
                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {Object.keys(detailedData.cleaned_data_preview[0] || {}).map(col => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {detailedData.cleaned_data_preview.map((row, idx) => (
                        <tr key={idx}>
                          {Object.values(row).map((value, colIdx) => (
                            <td key={colIdx}>{String(value)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={`message-bubble ${message.type}`}>
      <div className="message-content">
        <div 
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formatContent(message.content) }}
        />
        
        {/* Render detailed data if available */}
        {message.type === 'agent' && message.detailedData && renderDetailedData(message.detailedData)}
        
        <div className="message-time">
          {message.timestamp.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
          })}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;