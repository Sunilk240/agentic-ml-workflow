import { useNavigate } from 'react-router-dom';
import '../styles/userGuide.css';

const UserGuide = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/');
  };

  return (
    <div className="user-guide-page">
      <div className="guide-container">
        {/* Hero Section */}
        <div className="hero-section">
          <h1>ML Pipeline Assistant - User Guide</h1>
          <p className="hero-subtitle">
            Automated Machine Learning Made Simple
          </p>
          <p className="hero-description">
            Transform your data into insights with our intelligent ML pipeline that guides you through every step of the machine learning process.
          </p>
          <button className="get-started-btn" onClick={handleGetStarted}>
            🚀 Get Started
          </button>
        </div>

        {/* Pipeline Overview */}
        <section className="guide-section">
          <h2>🔄 Complete ML Pipeline</h2>
          <p>Our assistant follows a proven 4-phase approach:</p>
          <div className="pipeline-phases">
            <div className="phase-card">
              <div className="phase-icon">📊</div>
              <h3>Phase 1: Data Exploration</h3>
            </div>
            <div className="phase-card">
              <div className="phase-icon">🧹</div>
              <h3>Phase 2: Data Cleaning</h3>
            </div>
            <div className="phase-card">
              <div className="phase-icon">🔧</div>
              <h3>Phase 3: Feature Engineering</h3>
            </div>
            <div className="phase-card">
              <div className="phase-icon">🤖</div>
              <h3>Phase 4: Model Training/Clustering</h3>
            </div>
          </div>
        </section>

        {/* Phase 1: Data Exploration */}
        <section className="guide-section">
          <h2>📊 Phase 1: Data Exploration</h2>
          <div className="phase-content">
            <h3>What it does:</h3>
            <ul>
              <li>Automatically analyzes your dataset structure</li>
              <li>Identifies column types (numeric vs categorical)</li>
              <li>Detects data quality issues</li>
              <li>Provides comprehensive dataset overview</li>
            </ul>
            
            <h3>Features:</h3>
            <div className="features-grid">
              <div className="feature-item">✅ Dataset shape and memory usage analysis</div>
              <div className="feature-item">✅ Column type detection and statistics</div>
              <div className="feature-item">✅ Missing values and duplicate detection</div>
              <div className="feature-item">✅ Interactive data preview with first 5 rows</div>
              <div className="feature-item">✅ Automatic quality assessment</div>
            </div>

            <div className="commands-section">
              <h4>Commands to try:</h4>
              <div className="command-list">
                <code>"explore data"</code>
                <code>"show dataset info"</code>
              </div>
            </div>
          </div>
        </section>

        {/* Phase 2: Data Cleaning */}
        <section className="guide-section">
          <h2>🧹 Phase 2: Data Cleaning</h2>
          <div className="phase-content">
            <p className="section-subtitle">Intelligent Data Quality Fixes</p>
            
            <h3>What we detect & fix:</h3>
            
            <div className="cleaning-types">
              <div className="cleaning-type">
                <h4>🔍 Missing Values</h4>
                <p><strong>Detection:</strong> Automatic identification across all columns</p>
                <p><strong>Solutions:</strong> Fill with mean/median, forward fill, drop rows</p>
                <p><strong>Smart recommendations</strong> based on data type</p>
              </div>

              <div className="cleaning-type">
                <h4>📋 Duplicate Rows</h4>
                <p><strong>Detection:</strong> Exact duplicate identification</p>
                <p><strong>Solutions:</strong> Keep first/last occurrence, remove all</p>
                <p><strong>Impact analysis</strong> before removal</p>
              </div>

              <div className="cleaning-type">
                <h4>📈 Outliers</h4>
                <p><strong>Detection:</strong> IQR method (1.5 * IQR rule)</p>
                <p><strong>Solutions:</strong> Capping, removal, median replacement</p>
                <p><strong>Column-wise</strong> outlier analysis</p>
              </div>
            </div>

            <h3>Cleaning Approaches:</h3>
            <div className="features-grid">
              <div className="feature-item">✅ <strong>Recommended:</strong> AI-suggested optimal approach</div>
              <div className="feature-item">✅ <strong>Custom:</strong> Choose specific methods per issue</div>
              <div className="feature-item">✅ <strong>Before/After:</strong> Compare data quality metrics</div>
            </div>

            <div className="commands-section">
              <h4>Commands to try:</h4>
              <div className="command-list">
                <code>"perform cleaning"</code>
                <code>"proceed with cleaning"</code>
              </div>
            </div>
          </div>
        </section>

        {/* Phase 3: Feature Engineering */}
        <section className="guide-section">
          <h2>🔧 Phase 3: Feature Engineering</h2>
          <div className="phase-content">
            <p className="section-subtitle">Smart Feature Transformations</p>
            
            <div className="engineering-types">
              <div className="engineering-type">
                <h4>🎯 Categorical Encoding</h4>
                <h5>Techniques Supported:</h5>
                <ul>
                  <li><strong>One-Hot Encoding:</strong> Creates binary columns (recommended)</li>
                  <li><strong>Label Encoding:</strong> Ordinal numbering</li>
                  <li><strong>Frequency Encoding:</strong> Based on value counts</li>
                </ul>
                <p><strong>Best for:</strong> Converting text/categorical data to numbers</p>
              </div>

              <div className="engineering-type">
                <h4>📐 Feature Scaling</h4>
                <h5>Methods Available:</h5>
                <ul>
                  <li><strong>Standard Scaling:</strong> Mean=0, Std=1 (recommended)</li>
                  <li><strong>Min-Max Scaling:</strong> Scale to 0-1 range</li>
                  <li><strong>Robust Scaling:</strong> Median-based, outlier resistant</li>
                </ul>
                <p><strong>Best for:</strong> Normalizing different feature ranges</p>
              </div>

              <div className="engineering-type">
                <h4>🔗 Correlation Analysis</h4>
                <h5>Multicollinearity Solutions:</h5>
                <ul>
                  <li><strong>Remove Redundant:</strong> Drop highly correlated features</li>
                  <li><strong>PCA:</strong> Principal component analysis</li>
                  <li><strong>Manual Selection:</strong> Expert choice</li>
                </ul>
                <p><strong>Best for:</strong> Preventing feature redundancy</p>
              </div>
            </div>

            <h3>✨ What We Analyze:</h3>
            <div className="features-grid">
              <div className="feature-item">✅ Feature correlation matrix</div>
              <div className="feature-item">✅ Scaling requirements detection</div>
              <div className="feature-item">✅ High cardinality categorical features</div>
              <div className="feature-item">✅ Feature importance rankings</div>
            </div>

            <div className="commands-section">
              <h4>Commands to try:</h4>
              <div className="command-list">
                <code>"perform feature engineering"</code>
                <code>"proceed with feature engineering"</code>
              </div>
            </div>
          </div>
        </section>

        {/* Phase 4A: Model Training */}
        <section className="guide-section">
          <h2>🤖 Phase 4A: Model Training</h2>
          <div className="phase-content">
            <p className="section-subtitle">Automated Problem Detection & Algorithm Selection</p>
            
            <h3>🎯 Problem Type Detection</h3>
            <p>Our AI automatically detects:</p>
            <ul>
              <li><strong>Classification:</strong> Predicting categories (spam/not-spam, disease types)</li>
              <li><strong>Regression:</strong> Predicting numbers (prices, scores, quantities)</li>
            </ul>

            <h4>Detection based on:</h4>
            <div className="features-grid">
              <div className="feature-item">✅ Target column data type analysis</div>
              <div className="feature-item">✅ Unique value ratio calculation</div>
              <div className="feature-item">✅ Sample value examination</div>
              <div className="feature-item">✅ Statistical pattern recognition</div>
            </div>

            <h3>📚 Supported Algorithms</h3>
            
            <div className="algorithms-section">
              <div className="algorithm-category">
                <h4>For Classification Problems:</h4>
                <div className="algorithm-group">
                  <h5>🏆 Recommended Algorithms:</h5>
                  <ul>
                    <li><strong>Random Forest:</strong> Excellent for most datasets, handles overfitting</li>
                    <li><strong>Gradient Boosting:</strong> High performance, sequential learning</li>
                    <li><strong>Logistic Regression:</strong> Fast, interpretable baseline</li>
                  </ul>
                </div>
                <div className="algorithm-group">
                  <h5>🔄 Alternative Algorithms:</h5>
                  <ul>
                    <li><strong>SVM:</strong> Great for high-dimensional data</li>
                    <li><strong>K-Nearest Neighbors:</strong> Simple, good for local patterns</li>
                    <li><strong>Naive Bayes:</strong> Fast, works well with small datasets</li>
                  </ul>
                </div>
              </div>

              <div className="algorithm-category">
                <h4>For Regression Problems:</h4>
                <div className="algorithm-group">
                  <h5>🏆 Recommended Algorithms:</h5>
                  <ul>
                    <li><strong>Random Forest:</strong> Robust, handles non-linearity well</li>
                    <li><strong>Gradient Boosting:</strong> High performance predictive power</li>
                    <li><strong>Linear Regression:</strong> Simple, interpretable baseline</li>
                  </ul>
                </div>
                <div className="algorithm-group">
                  <h5>🔄 Alternative Algorithms:</h5>
                  <ul>
                    <li><strong>Ridge Regression:</strong> Regularized, prevents overfitting</li>
                    <li><strong>Lasso Regression:</strong> Feature selection + regularization</li>
                    <li><strong>SVR:</strong> Support Vector Regression</li>
                    <li><strong>K-Nearest Neighbors:</strong> Non-parametric approach</li>
                  </ul>
                </div>
              </div>
            </div>

            <h3>📊 What You Get:</h3>
            <div className="features-grid">
              <div className="feature-item">✅ <strong>Performance Metrics:</strong> Accuracy, Precision, Recall, F1-Score (Classification) or R², MSE, MAE (Regression)</div>
              <div className="feature-item">✅ <strong>Cross-Validation:</strong> 5-fold CV scores for reliability</div>
              <div className="feature-item">✅ <strong>Feature Importance:</strong> Which features matter most</div>
              <div className="feature-item">✅ <strong>Model Comparison:</strong> When multiple algorithms trained</div>
              <div className="feature-item">✅ <strong>Performance Interpretation:</strong> Easy-to-understand results</div>
            </div>

            <div className="commands-section">
              <h4>Commands to try:</h4>
              <div className="command-list">
                <code>"train model"</code>
                <code>"proceed with training random_forest"</code>
                <code>"proceed with training recommended"</code>
              </div>
            </div>
          </div>
        </section>

        {/* Phase 4B: Clustering */}
        <section className="guide-section">
          <h2>🎯 Phase 4B: Clustering Analysis</h2>
          <div className="phase-content">
            <p className="section-subtitle">Discover Hidden Patterns in Your Data</p>
            
            <h3>🔍 When to Use Clustering:</h3>
            <ul>
              <li>No clear target variable to predict</li>
              <li>Want to find natural groups in data</li>
              <li>Customer segmentation analysis</li>
              <li>Pattern discovery in unlabeled data</li>
            </ul>

            <h3>🤖 Supported Algorithms</h3>
            
            <div className="clustering-algorithms">
              <div className="clustering-algorithm">
                <h4>📊 K-Means</h4>
                <p><strong>Best for:</strong> Spherical, well-separated clusters</p>
                <p><strong>Pros:</strong> Fast, scalable, interpretable</p>
                <p><strong>Cons:</strong> Requires cluster count, assumes spherical shapes</p>
                <p><strong>Use when:</strong> You have large datasets with compact clusters</p>
              </div>

              <div className="clustering-algorithm">
                <h4>🌳 Hierarchical</h4>
                <p><strong>Best for:</strong> Understanding cluster relationships</p>
                <p><strong>Pros:</strong> No cluster count needed, creates dendrograms</p>
                <p><strong>Cons:</strong> Slow on large datasets, memory intensive</p>
                <p><strong>Use when:</strong> You want cluster hierarchy visualization</p>
              </div>

              <div className="clustering-algorithm">
                <h4>🎯 DBSCAN</h4>
                <p><strong>Best for:</strong> Irregular shapes, noisy data</p>
                <p><strong>Pros:</strong> Finds arbitrary shapes, handles outliers</p>
                <p><strong>Cons:</strong> Sensitive to parameters, struggles with varying densities</p>
                <p><strong>Use when:</strong> You have irregularly shaped clusters</p>
              </div>

              <div className="clustering-algorithm">
                <h4>📈 Gaussian Mixture</h4>
                <p><strong>Best for:</strong> Overlapping clusters, probabilistic assignments</p>
                <p><strong>Pros:</strong> Soft clustering, uncertainty estimates</p>
                <p><strong>Cons:</strong> Assumes gaussian distributions, computationally intensive</p>
                <p><strong>Use when:</strong> Clusters may overlap or you need probabilities</p>
              </div>
            </div>

            <h3>🎨 What You Get:</h3>
            <div className="features-grid">
              <div className="feature-item">✅ <strong>Optimal Cluster Count:</strong> AI-suggested using elbow method</div>
              <div className="feature-item">✅ <strong>Cluster Quality Metrics:</strong> Silhouette score, Calinski-Harabasz index</div>
              <div className="feature-item">✅ <strong>Cluster Characteristics:</strong> Size, feature means, distinguishing traits</div>
              <div className="feature-item">✅ <strong>Algorithm Comparison:</strong> Performance across different methods</div>
              <div className="feature-item">✅ <strong>Data Suitability:</strong> Assessment of clustering potential</div>
            </div>

            <div className="commands-section">
              <h4>Commands to try:</h4>
              <div className="command-list">
                <code>"perform clustering"</code>
                <code>"execute clustering kmeans"</code>
                <code>"execute clustering recommended"</code>
              </div>
            </div>
          </div>
        </section>

        {/* Getting Started */}
        <section className="guide-section getting-started">
          <h2>🚀 Getting Started</h2>
          <div className="phase-content">
            <h3>Step-by-Step Workflow:</h3>
            <div className="workflow-steps">
              <div className="workflow-step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h4>📤 Upload Dataset</h4>
                  <p>CSV files supported</p>
                </div>
              </div>
              <div className="workflow-step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h4>📊 Explore</h4>
                  <p>"explore data" → Get dataset overview</p>
                </div>
              </div>
              <div className="workflow-step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h4>🧹 Clean</h4>
                  <p>"perform cleaning" → Fix data quality issues</p>
                </div>
              </div>
              <div className="workflow-step">
                <div className="step-number">4</div>
                <div className="step-content">
                  <h4>🔧 Engineer</h4>
                  <p>"perform feature engineering" → Transform features</p>
                </div>
              </div>
              <div className="workflow-step">
                <div className="step-number">5</div>
                <div className="step-content">
                  <h4>🤖 Train OR 🎯 Cluster</h4>
                  <p>"train model" → Build ML models<br/>OR<br/>"perform clustering" → Find data patterns</p>
                </div>
              </div>
            </div>

            <h3>💡 Example Commands:</h3>
            <div className="example-commands">
              <code>explore data</code>
              <code>perform cleaning</code>
              <code>proceed with cleaning</code>
              <code>perform feature engineering</code>
              <code>proceed with feature engineering</code>
              <code>train model</code>
              <code>proceed with training recommended</code>
            </div>

            <div className="final-cta">
              <h3>🎯 Ready to Start?</h3>
              <p>Upload your dataset and begin your ML journey with our intelligent assistant!</p>
              <button className="launch-btn" onClick={handleGetStarted}>
                🚀 Launch ML Assistant
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default UserGuide;
