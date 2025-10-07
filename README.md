# 🤖 AI-Powered ML Pipeline Platform

A comprehensive, educational machine learning platform that transforms the entire ML workflow experience with rich visualizations, intelligent recommendations, and professional-grade analysis tools.

![ML Pipeline](https://img.shields.io/badge/ML-Pipeline-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![React](https://img.shields.io/badge/React-19.1+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

### 🎯 **Complete ML Workflow Coverage**
- **📊 Data Exploration** → Rich data insights and quality assessment
- **🧹 Data Cleaning** → Automated cleaning with transparency
- **🔧 Feature Engineering** → Smart feature transformations
- **🤖 Model Training** → Comprehensive model evaluation
- **🎯 Clustering** → Unsupervised learning with cluster insights

### 🎨 **Rich Frontend Visualizations**
- Interactive data tables and quality metrics
- Performance charts and feature importance
- Color-coded indicators for quality and status
- Detailed diagnostics and recommendations
- Professional summaries with actionable insights
- Algorithm comparisons with pros/cons
- Visualization-ready data for cluster plots

### 🧠 **AI-Powered Intelligence**
- **Smart Algorithm Recommendations** based on data characteristics
- **Automated Quality Assessment** with multiple metrics
- **Educational Insights** about ML concepts and best practices
- **Business-Friendly Interpretations** of technical results
- **Proactive Issue Detection** and resolution suggestions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd ml-pipeline-platform
```

2. **Set up Python environment**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys (Gemini, Groq, etc.)
```

5. **Start the backend server**
```bash
python app.py
```
Backend will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Start the development server**
```bash
npm run dev
```
Frontend will be available at `http://localhost:5173`

## 📁 Project Structure

```
ml-pipeline-platform/
├── backend/                    # FastAPI backend
│   ├── app.py                 # Main application entry point
│   ├── config.py              # Configuration settings
│   ├── helpers.py             # Utility functions
│   ├── data_exploration.py    # Data analysis tools
│   ├── data_cleaning.py       # Data cleaning algorithms
│   ├── feature_engineering.py # Feature transformation tools
│   ├── model_training.py      # ML model training
│   ├── clustering.py          # Clustering algorithms
│   ├── model_evaluation.py    # Model assessment tools
│   ├── model_comparison.py    # Model comparison utilities
│   ├── problem_detection.py   # Problem type detection
│   ├── session_management.py  # Session handling
│   ├── uploads/               # File upload directory
│   └── requirements.txt       # Python dependencies
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── FeatureEngineeringDisplay.jsx
│   │   │   ├── ModelTrainingDisplay.jsx
│   │   │   ├── ClusteringDisplay.jsx
│   │   │   └── ...
│   │   ├── styles/           # CSS styles
│   │   └── App.jsx           # Main app component
│   ├── package.json          # Node.js dependencies
│   └── vite.config.js        # Vite configuration
└── README.md                 # This file
```

## 🎯 Usage Guide

### 1. **Upload Your Dataset**
- Drag and drop CSV files or click to upload
- Supports various data formats and sizes
- Automatic data type detection and validation

### 2. **Explore Your Data**
```
Type: "explore dataset"
```
- Get comprehensive data overview
- Quality assessment and missing value analysis
- Column statistics and data type information
- Interactive data preview tables

### 3. **Clean Your Data**
```
Type: "perform data cleaning"  # Analysis phase
Type: "execute data cleaning"  # Execution phase
```
- Automated issue detection (missing values, duplicates, outliers)
- Severity assessment and cleaning recommendations
- Before/after comparisons with quality improvements

### 4. **Engineer Features**
```
Type: "perform feature engineering"  # Analysis phase
Type: "execute feature engineering"  # Execution phase
```
- Feature type analysis and correlation detection
- Scaling and transformation recommendations
- Feature importance and relationship analysis

### 5. **Train Models**
```
Type: "train model"           # Analysis phase
Type: "execute model training" # Execution phase
```
- Automatic problem type detection (classification/regression)
- Algorithm recommendations based on data characteristics
- Comprehensive performance metrics and cross-validation

### 6. **Perform Clustering**
```
Type: "perform clustering"    # Analysis phase
Type: "execute clustering kmeans" # Execution phase
```
- Clustering suitability assessment
- Algorithm comparison (K-Means, Hierarchical, DBSCAN, Gaussian Mixture)
- Quality metrics and cluster interpretation

## 🔧 Technical Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **LangChain** - Framework for developing LLM applications
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization

### Frontend
- **React 19** - Modern UI library
- **Vite** - Fast build tool and development server
- **Axios** - HTTP client for API communication
- **CSS3** - Custom styling with responsive design

### AI Integration
- **Google Gemini** - Advanced language model for intelligent recommendations
- **Groq** - High-performance inference for fast responses

## 🎨 Key Components

### Data Analysis Components
- **DataExploration** - Interactive data overview and quality metrics
- **DataCleaning** - Issue detection and cleaning recommendations
- **FeatureEngineering** - Feature analysis and transformation tools

### Model Training Components
- **ModelTraining** - Algorithm recommendations and training analysis
- **ModelResults** - Performance metrics and evaluation dashboards
- **ModelComparison** - Side-by-side algorithm comparison

### Clustering Components
- **ClusteringDisplay** - Algorithm suitability and recommendations
- **ClusteringResults** - Cluster analysis and quality metrics
- **ClusterVisualization** - Interactive cluster plots and insights

## 📊 Supported Algorithms

### Supervised Learning
- **Classification**: Logistic Regression, Random Forest, SVM, Gradient Boosting
- **Regression**: Linear Regression, Random Forest, SVR, Gradient Boosting

### Unsupervised Learning
- **Clustering**: K-Means, Hierarchical, DBSCAN, Gaussian Mixture Models

### Feature Engineering
- **Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Encoding**: One-Hot Encoding, Label Encoding
- **Transformation**: Log, Square Root, Polynomial Features

## 🔍 Quality Metrics

### Model Evaluation
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: MAE, MSE, RMSE, R², Cross-validation scores

### Clustering Quality
- **Silhouette Score** - Cluster separation quality
- **Calinski-Harabasz Index** - Cluster density ratio
- **Davies-Bouldin Index** - Cluster similarity measure
- **Inertia** - Within-cluster sum of squares

## 🚀 Advanced Features

### Intelligent Recommendations
- **Algorithm Selection** based on data characteristics
- **Hyperparameter Suggestions** for optimal performance
- **Feature Engineering** recommendations
- **Data Quality** improvement suggestions

### Educational Insights
- **Algorithm Explanations** with pros/cons
- **Performance Interpretations** in business terms
- **Best Practices** and ML concepts
- **Troubleshooting** guides and tips

### Professional Visualizations
- **Interactive Charts** and performance dashboards
- **Quality Indicators** with color-coded status
- **Comparison Tables** for algorithm selection
- **Export-Ready** results and summaries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn** community for excellent ML algorithms
- **React** team for the amazing frontend framework
- **FastAPI** for the high-performance backend framework
- **LangChain** for LLM integration capabilities

---

**Built with ❤️ for the ML community**

Transform your data science workflow with intelligent automation, rich visualizations, and educational insights that make machine learning accessible to everyone.