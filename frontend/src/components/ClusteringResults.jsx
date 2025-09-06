import React from 'react';

const ClusteringResults = ({ data }) => {
  if (!data || !data.clustering_results) return null;

  const { 
    clustering_results, 
    cluster_statistics, 
    quality_metrics,
    cluster_characteristics,
    visualization_data,
    algorithm_comparison,
    data_summary
  } = data;

  const getQualityColor = (level) => {
    switch(level) {
      case 'excellent': return 'excellent';
      case 'good': return 'good';
      case 'fair': return 'fair';
      case 'poor': return 'poor';
      default: return 'fair';
    }
  };

  const getClusterColor = (index) => {
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'];
    return colors[index % colors.length];
  };

  return (
    <div className="clustering-results">
      {/* Results Header */}
      <div className="results-header">
        <h4>🎉 Clustering Complete</h4>
        <div className="clustering-info">
          <span className="algorithm-name">{clustering_results.algorithm_display_name}</span>
          <span className="cluster-count">{clustering_results.n_clusters} Clusters</span>
        </div>
      </div>

      {/* Key Metrics Overview */}
      <div className="metrics-overview">
        <div className="primary-metric">
          <div className="metric-card main-metric">
            <span className="metric-value">{quality_metrics.silhouette_score.toFixed(3)}</span>
            <span className="metric-label">SILHOUETTE SCORE</span>
            <span className={`quality-badge ${getQualityColor(quality_metrics.quality_level)}`}>
              {quality_metrics.interpretation.toUpperCase()}
            </span>
          </div>
        </div>

        <div className="secondary-metrics">
          <div className="metric-card">
            <span className="metric-value">{clustering_results.n_clusters}</span>
            <span className="metric-label">CLUSTERS</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">{clustering_results.total_samples.toLocaleString()}</span>
            <span className="metric-label">SAMPLES</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">{quality_metrics.calinski_harabasz.toFixed(1)}</span>
            <span className="metric-label">CALINSKI-HARABASZ</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">{quality_metrics.davies_bouldin.toFixed(3)}</span>
            <span className="metric-label">DAVIES-BOULDIN</span>
          </div>
        </div>
      </div>

      {/* Cluster Distribution */}
      <div className="cluster-distribution-section">
        <h5>📊 Cluster Distribution</h5>
        <div className="distribution-overview">
          <div className="distribution-stats">
            <div className="stat-item">
              <span className="stat-label">Balance:</span>
              <span className={`stat-value ${cluster_statistics.size_balance === 'well-balanced' ? 'balanced' : 'imbalanced'}`}>
                {cluster_statistics.size_balance.replace('-', ' ').toUpperCase()}
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Largest:</span>
              <span className="stat-value">Cluster {cluster_statistics.largest_cluster}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Smallest:</span>
              <span className="stat-value">Cluster {cluster_statistics.smallest_cluster}</span>
            </div>
          </div>

          {/* Cluster Size Bars */}
          <div className="cluster-size-bars">
            {cluster_statistics.cluster_sizes.map((size, index) => (
              <div key={index} className="cluster-bar">
                <div className="bar-info">
                  <span className="cluster-label">Cluster {index}</span>
                  <span className="cluster-size">{size} ({cluster_statistics.cluster_percentages[index].toFixed(1)}%)</span>
                </div>
                <div className="bar-container">
                  <div 
                    className="bar-fill"
                    style={{ 
                      width: `${cluster_statistics.cluster_percentages[index]}%`,
                      backgroundColor: getClusterColor(index)
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Quality Metrics Detail */}
      <div className="quality-metrics-section">
        <h5>📈 Quality Metrics</h5>
        <div className="quality-grid">
          <div className="quality-card">
            <h6>🎯 Silhouette Score</h6>
            <div className="quality-info">
              <span className="quality-value">{quality_metrics.silhouette_score.toFixed(3)}</span>
              <span className="quality-description">
                Measures how well-separated clusters are (-1 to 1, higher is better)
              </span>
            </div>
          </div>

          <div className="quality-card">
            <h6>📊 Calinski-Harabasz</h6>
            <div className="quality-info">
              <span className="quality-value">{quality_metrics.calinski_harabasz.toFixed(1)}</span>
              <span className="quality-description">
                Ratio of between-cluster to within-cluster dispersion (higher is better)
              </span>
            </div>
          </div>

          <div className="quality-card">
            <h6>🔗 Davies-Bouldin</h6>
            <div className="quality-info">
              <span className="quality-value">{quality_metrics.davies_bouldin.toFixed(3)}</span>
              <span className="quality-description">
                Average similarity between clusters (lower is better)
              </span>
            </div>
          </div>

          {quality_metrics.inertia > 0 && (
            <div className="quality-card">
              <h6>⚡ Inertia</h6>
              <div className="quality-info">
                <span className="quality-value">{quality_metrics.inertia.toFixed(1)}</span>
                <span className="quality-description">
                  Within-cluster sum of squares (lower is better)
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Cluster Characteristics */}
      <div className="cluster-characteristics-section">
        <h5>🔍 Cluster Characteristics</h5>
        <div className="characteristics-grid">
          {Object.entries(cluster_characteristics).map(([clusterId, characteristics], index) => (
            <div key={clusterId} className="cluster-card">
              <div className="cluster-header">
                <div className="cluster-title">
                  <div 
                    className="cluster-color-indicator"
                    style={{ backgroundColor: getClusterColor(index) }}
                  ></div>
                  <h6>Cluster {clusterId.split('_')[1]}</h6>
                </div>
                <div className="cluster-size-info">
                  <span className="size-count">{characteristics.size}</span>
                  <span className="size-percentage">({characteristics.percentage.toFixed(1)}%)</span>
                </div>
              </div>

              <div className="cluster-description">
                <p>{characteristics.description}</p>
              </div>

              <div className="key-features">
                <h6>🎯 Key Features</h6>
                <div className="feature-levels">
                  {Object.entries(characteristics.key_features).slice(0, 4).map(([feature, level]) => (
                    <div key={feature} className="feature-level">
                      <span className="feature-name">{feature}</span>
                      <span className={`level-badge ${level}`}>{level.toUpperCase()}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="centroid-info">
                <h6>📍 Centroid</h6>
                <div className="centroid-values">
                  {characteristics.centroid.slice(0, 3).map((value, idx) => (
                    <span key={idx} className="centroid-value">
                      {value.toFixed(2)}
                    </span>
                  ))}
                  {characteristics.centroid.length > 3 && (
                    <span className="more-values">+{characteristics.centroid.length - 3} more</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Visualization Data */}
      {visualization_data && (visualization_data.pca_data || visualization_data.scatter_data) && (
        <div className="visualization-section">
          <h5>📊 Cluster Visualization</h5>
          
          {visualization_data.pca_data && (
            <div className="pca-info">
              <div className="pca-stats">
                <div className="pca-stat">
                  <span className="pca-label">Total Variance Explained:</span>
                  <span className="pca-value">
                    {(visualization_data.pca_data.total_variance_explained * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="pca-components">
                  <span className="component-info">
                    PC1: {(visualization_data.pca_data.explained_variance[0] * 100).toFixed(1)}%
                  </span>
                  <span className="component-info">
                    PC2: {(visualization_data.pca_data.explained_variance[1] * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              
              <div className="visualization-note">
                <p>📊 2D visualization using Principal Component Analysis (PCA) to reduce dimensionality while preserving cluster structure.</p>
              </div>
            </div>
          )}

          {visualization_data.scatter_data && (
            <div className="scatter-info">
              <div className="scatter-axes">
                <span className="axis-info">X-axis: {visualization_data.scatter_data.x_feature}</span>
                <span className="axis-info">Y-axis: {visualization_data.scatter_data.y_feature}</span>
              </div>
            </div>
          )}

          {/* Cluster Centers */}
          {visualization_data.cluster_centers && (
            <div className="cluster-centers-info">
              <h6>📍 Cluster Centers</h6>
              <div className="centers-grid">
                {visualization_data.cluster_centers.x_centers.map((x, index) => (
                  <div key={index} className="center-point">
                    <div 
                      className="center-indicator"
                      style={{ backgroundColor: getClusterColor(index) }}
                    ></div>
                    <span className="center-coords">
                      Cluster {index}: ({x.toFixed(2)}, {visualization_data.cluster_centers.y_centers[index].toFixed(2)})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Algorithm Comparison */}
      {Object.keys(algorithm_comparison).length > 1 && (
        <div className="algorithm-comparison-section">
          <h5>🔄 Algorithm Comparison</h5>
          <div className="comparison-table">
            <table>
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>Clusters</th>
                  <th>Silhouette</th>
                  <th>Calinski-Harabasz</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(algorithm_comparison).map(([algName, metrics]) => (
                  <tr key={algName} className={algName === clustering_results.algorithm_used ? 'selected' : ''}>
                    <td className="algorithm-name">{algName.replace('_', ' ').toUpperCase()}</td>
                    <td>{metrics.n_clusters}</td>
                    <td>{metrics.silhouette_score.toFixed(3)}</td>
                    <td>{metrics.calinski_harabasz.toFixed(1)}</td>
                    <td>
                      {algName === clustering_results.algorithm_used ? (
                        <span className="status-badge selected">👑 SELECTED</span>
                      ) : (
                        <span className="status-badge alternative">Alternative</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Clustering Summary */}
      <div className="clustering-summary-section">
        <h5>📋 Clustering Summary</h5>
        <div className="summary-grid">
          <div className="summary-item">
            <span className="summary-label">Algorithm:</span>
            <span className="summary-value">{clustering_results.algorithm_display_name}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Execution Time:</span>
            <span className="summary-value">{clustering_results.clustering_time}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Convergence:</span>
            <span className="summary-value">
              {clustering_results.convergence ? '✅ Converged' : '❌ Did not converge'}
            </span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Features Used:</span>
            <span className="summary-value">{clustering_results.features_used.length}</span>
          </div>
        </div>

        <div className="preprocessing-info">
          <h6>🔧 Preprocessing Applied</h6>
          <div className="preprocessing-tags">
            {data_summary.preprocessing_applied.map((process, index) => (
              <span key={index} className="preprocessing-tag">{process}</span>
            ))}
          </div>
        </div>

        <div className="features-used">
          <h6>📊 Features Used</h6>
          <div className="feature-tags">
            {clustering_results.features_used.slice(0, 6).map((feature, index) => (
              <span key={index} className="feature-tag">{feature}</span>
            ))}
            {clustering_results.features_used.length > 6 && (
              <span className="more-features">+{clustering_results.features_used.length - 6} more</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClusteringResults;