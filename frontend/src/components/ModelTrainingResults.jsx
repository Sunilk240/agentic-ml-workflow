import React from 'react';

const ModelTrainingResults = ({ data }) => {
    if (!data || !data.training_summary) return null;

    const {
        training_summary,
        performance_metrics,
        feature_importance,
        model_diagnostics,
        predictions_sample,
        training_data_info
    } = data;

    const getPerformanceColor = (interpretation) => {
        switch (interpretation) {
            case 'excellent': return 'excellent';
            case 'very': return 'very-good';
            case 'good': return 'good';
            default: return 'needs-improvement';
        }
    };

    const getOverfittingStatus = (status) => {
        return status === 'good' ? 'good' : 'warning';
    };

    return (
        <div className="model-training-results">
            {/* Training Summary Header */}
            <div className="results-header">
                <h4>🎉 Model Training Complete</h4>
                <div className="model-info">
                    <span className="model-name">{training_summary.algorithm_display_name}</span>
                    <span className="problem-type">{training_summary.problem_type}</span>
                </div>
            </div>

            {/* Key Performance Metrics */}
            <div className="performance-overview">
                <div className="primary-metric">
                    <div className="metric-card main-metric">
                        <span className="metric-value">{performance_metrics.primary_metric.value.toFixed(4)}</span>
                        <span className="metric-label">{performance_metrics.primary_metric.name.replace('_', ' ').toUpperCase()}</span>
                        <span className={`performance-badge ${getPerformanceColor(performance_metrics.primary_metric.interpretation)}`}>
                            {performance_metrics.primary_metric.interpretation.toUpperCase()}
                        </span>
                    </div>
                </div>

                {/* Detailed Metrics Grid */}
                {performance_metrics.detailed_metrics && (
                    <div className="detailed-metrics-grid">
                        {Object.entries(performance_metrics.detailed_metrics).map(([metric, value]) => (
                            <div key={metric} className="metric-card">
                                <span className="metric-value">{value.toFixed(4)}</span>
                                <span className="metric-label">{metric.replace('_', ' ').toUpperCase()}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Cross Validation Results */}
            {performance_metrics.cross_validation && (
                <div className="cross-validation-section">
                    <h5>🔄 Cross Validation Results</h5>
                    <div className="cv-summary">
                        <div className="cv-stats">
                            <div className="cv-stat">
                                <span className="cv-label">Mean Score:</span>
                                <span className="cv-value">{performance_metrics.cross_validation.mean_score.toFixed(4)}</span>
                            </div>
                            <div className="cv-stat">
                                <span className="cv-label">Std Deviation:</span>
                                <span className="cv-value">±{performance_metrics.cross_validation.std_score.toFixed(4)}</span>
                            </div>
                            <div className="cv-stat">
                                <span className="cv-label">Confidence Interval:</span>
                                <span className="cv-value">
                                    [{performance_metrics.cross_validation.confidence_interval[0].toFixed(3)}, {performance_metrics.cross_validation.confidence_interval[1].toFixed(3)}]
                                </span>
                            </div>
                        </div>
                        <div className="cv-scores">
                            <h6>Individual Fold Scores:</h6>
                            <div className="fold-scores">
                                {performance_metrics.cross_validation.scores.map((score, index) => (
                                    <span key={index} className="fold-score">
                                        Fold {index + 1}: {score.toFixed(4)}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Feature Importance */}
            {feature_importance.available && (
                <div className="feature-importance-section">
                    <h5>🎯 Feature Importance</h5>
                    <div className="importance-info">
                        <p>Top {feature_importance.top_features.length} most important features out of {feature_importance.total_features} total features:</p>
                    </div>
                    <div className="importance-chart">
                        {feature_importance.top_features.map((feature, index) => (
                            <div key={feature.name} className="importance-bar">
                                <div className="feature-info">
                                    <span className="feature-rank">#{feature.rank}</span>
                                    <span className="feature-name">{feature.name}</span>
                                    <span className="feature-percentage">{feature.percentage.toFixed(1)}%</span>
                                </div>
                                <div className="importance-bar-container">
                                    <div
                                        className="importance-bar-fill"
                                        style={{ width: `${feature.percentage}%` }}
                                    ></div>
                                </div>
                                <span className="importance-value">{feature.importance.toFixed(4)}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Model Diagnostics */}
            <div className="model-diagnostics-section">
                <h5>🔍 Model Diagnostics</h5>

                {/* Overfitting Check */}
                <div className="diagnostic-card">
                    <h6>📊 Overfitting Analysis</h6>
                    <div className="overfitting-analysis">
                        <div className="score-comparison">
                            <div className="score-item">
                                <span className="score-label">Training Score (CV):</span>
                                <span className="score-value">{model_diagnostics.overfitting_check.train_score.toFixed(4)}</span>
                            </div>
                            <div className="score-item">
                                <span className="score-label">Test Score:</span>
                                <span className="score-value">{model_diagnostics.overfitting_check.test_score.toFixed(4)}</span>
                            </div>
                            <div className="score-item">
                                <span className="score-label">Difference:</span>
                                <span className="score-value">{model_diagnostics.overfitting_check.difference.toFixed(4)}</span>
                            </div>
                        </div>
                        <div className={`overfitting-status ${getOverfittingStatus(model_diagnostics.overfitting_check.status)}`}>
                            {model_diagnostics.overfitting_check.status === 'good' ?
                                '✅ No signs of overfitting' :
                                '⚠️ Potential overfitting detected'
                            }
                        </div>
                    </div>
                </div>

                {/* Prediction Confidence */}
                {model_diagnostics.prediction_confidence.confidence_distribution && (
                    <div className="diagnostic-card">
                        <h6>🎯 Prediction Confidence</h6>
                        <div className="confidence-analysis">
                            <div className="confidence-summary">
                                <span className="confidence-label">Mean Confidence:</span>
                                <span className="confidence-value">
                                    {(model_diagnostics.prediction_confidence.mean_confidence * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="confidence-distribution">
                                <div className="confidence-bar">
                                    <span className="conf-label">Very High (&gt;90%):</span>
                                    <span className="conf-count">{model_diagnostics.prediction_confidence.confidence_distribution.very_high}</span>
                                </div>
                                <div className="confidence-bar">
                                    <span className="conf-label">High (80-90%):</span>
                                    <span className="conf-count">{model_diagnostics.prediction_confidence.confidence_distribution.high}</span>
                                </div>
                                <div className="confidence-bar">
                                    <span className="conf-label">Medium (60-80%):</span>
                                    <span className="conf-count">{model_diagnostics.prediction_confidence.confidence_distribution.medium}</span>
                                </div>
                                <div className="confidence-bar">
                                    <span className="conf-label">Low (&lt;60%):</span>
                                    <span className="conf-count">{model_diagnostics.prediction_confidence.confidence_distribution.low}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Training Summary */}
            <div className="training-summary-section">
                <h5>📋 Training Summary</h5>
                <div className="summary-grid">
                    <div className="summary-item">
                        <span className="summary-label">Algorithm:</span>
                        <span className="summary-value">{training_summary.algorithm_display_name}</span>
                    </div>
                    <div className="summary-item">
                        <span className="summary-label">Training Samples:</span>
                        <span className="summary-value">{training_summary.training_samples.toLocaleString()}</span>
                    </div>
                    <div className="summary-item">
                        <span className="summary-label">Test Samples:</span>
                        <span className="summary-value">{training_summary.test_samples.toLocaleString()}</span>
                    </div>
                    <div className="summary-item">
                        <span className="summary-label">Features Used:</span>
                        <span className="summary-value">{training_summary.feature_count}</span>
                    </div>
                    <div className="summary-item">
                        <span className="summary-label">Model Complexity:</span>
                        <span className={`summary-value complexity-${training_summary.model_complexity}`}>
                            {training_summary.model_complexity.toUpperCase()}
                        </span>
                    </div>
                    <div className="summary-item">
                        <span className="summary-label">Cross Validation:</span>
                        <span className="summary-value">
                            {training_summary.cross_validation_performed ? '✅ Performed' : '❌ Not performed'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Hyperparameters */}
            {Object.keys(training_summary.hyperparameters_used).length > 0 && (
                <div className="hyperparameters-section">
                    <h5>⚙️ Model Hyperparameters</h5>
                    <div className="hyperparameters-grid">
                        {Object.entries(training_summary.hyperparameters_used).slice(0, 8).map(([param, value]) => (
                            <div key={param} className="hyperparam-item">
                                <span className="param-name">{param.replace('_', ' ')}:</span>
                                <span className="param-value">{String(value)}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Confusion Matrix for Classification */}
            {performance_metrics.confusion_matrix && (
                <div className="confusion-matrix-section">
                    <h5>📊 Confusion Matrix</h5>
                    <div className="confusion-matrix">
                        <table className="confusion-table">
                            <thead>
                                <tr>
                                    <th></th>
                                    {performance_metrics.confusion_matrix[0].map((_, index) => (
                                        <th key={index}>Pred {index}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {performance_metrics.confusion_matrix.map((row, rowIndex) => (
                                    <tr key={rowIndex}>
                                        <th>Actual {rowIndex}</th>
                                        {row.map((value, colIndex) => (
                                            <td key={colIndex} className={rowIndex === colIndex ? 'diagonal' : 'off-diagonal'}>
                                                {value}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Predictions Sample */}
            {predictions_sample.actual_values && predictions_sample.predicted_values && (
                <div className="predictions-sample-section">
                    <h5>🔮 Sample Predictions</h5>
                    <div className="predictions-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Sample</th>
                                    <th>Actual</th>
                                    <th>Predicted</th>
                                    <th>Match</th>
                                </tr>
                            </thead>
                            <tbody>
                                {predictions_sample.actual_values.slice(0, 5).map((actual, index) => {
                                    const predicted = predictions_sample.predicted_values[index];
                                    const isMatch = training_summary.problem_type === 'classification' ?
                                        actual === predicted :
                                        Math.abs(actual - predicted) < Math.abs(actual * 0.1);

                                    return (
                                        <tr key={index}>
                                            <td>#{index + 1}</td>
                                            <td>{typeof actual === 'number' ? actual.toFixed(4) : actual}</td>
                                            <td>{typeof predicted === 'number' ? predicted.toFixed(4) : predicted}</td>
                                            <td className={isMatch ? 'match' : 'no-match'}>
                                                {isMatch ? '✅' : '❌'}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Training Data Info */}
            <div className="training-data-info">
                <h5>📈 Dataset Information</h5>
                <div className="data-info-grid">
                    <div className="info-item">
                        <span className="info-label">Total Samples:</span>
                        <span className="info-value">{training_data_info.total_samples.toLocaleString()}</span>
                    </div>
                    <div className="info-item">
                        <span className="info-label">Train/Test Split:</span>
                        <span className="info-value">
                            {((training_data_info.training_samples / training_data_info.total_samples) * 100).toFixed(0)}% / {((training_data_info.test_samples / training_data_info.total_samples) * 100).toFixed(0)}%
                        </span>
                    </div>
                    <div className="info-item">
                        <span className="info-label">Features:</span>
                        <span className="info-value">{training_data_info.feature_count}</span>
                    </div>
                    <div className="info-item">
                        <span className="info-label">Target:</span>
                        <span className="info-value">{training_summary.target_column}</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ModelTrainingResults;