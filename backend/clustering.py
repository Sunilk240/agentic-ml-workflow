import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from helpers import DATAFRAME_CACHE, PIPELINE_STATE, get_dataset
from config import Config

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

@tool
def perform_clustering(path: str) -> Dict[str, Any]:
    """
    Shows available clustering algorithms and their descriptions.
    User says 'perform clustering' or 'cluster data'.
    """
    df = get_dataset(path)
    if df is None:
        return {"error": "Dataset not found"}
    
    # Check if we have enough numeric features
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric features for clustering"}
    
    # Format response showing available algorithms
    response = "🎯 **Available Clustering Algorithms**\n\n"
    response += f"**Dataset:** {len(df)} samples, {len(numeric_cols)} numeric features\n"
    response += f"**Features:** {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}\n\n"
    
    response += "**🔄 CLUSTERING ALGORITHMS:**\n"
    response += "• **K-Means** - Good for spherical, well-separated clusters (requires number of clusters)\n"
    response += "• **Hierarchical** - Creates tree-like structure, good for nested relationships\n"
    response += "• **DBSCAN** - Density-based, finds irregular shapes and handles noise\n"
    response += "• **Gaussian Mixture** - Probabilistic, handles overlapping clusters\n\n"
    
    response += "**💡 NEXT STEPS:**\n"
    response += "• Type 'execute clustering kmeans' for K-Means clustering\n"
    response += "• Type 'execute clustering hierarchical' for Hierarchical clustering\n"
    response += "• Type 'execute clustering dbscan' for DBSCAN clustering\n"
    response += "• Type 'execute clustering gaussian_mixture' for Gaussian Mixture\n"
    response += "• Type 'execute clustering recommended' to run K-Means and Hierarchical\n"
    
    # Collect detailed data for frontend
    from sklearn.decomposition import PCA
    
    # Dataset characteristics analysis
    sample_size = len(df)
    feature_count = len(numeric_cols)
    categorical_features = len(df.select_dtypes(include=['object']).columns)
    
    # Analyze data suitability for clustering
    X_sample = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Calculate data density and dimensionality
    data_density = "dense" if X_sample.isnull().sum().sum() / (len(X_sample) * len(numeric_cols)) < 0.1 else "sparse"
    dimensionality = "high" if feature_count > 10 else "medium" if feature_count > 5 else "low"
    
    # Clustering suitability analysis
    clustering_suitability_score = 0.5  # Base score
    suitability_reasons = []
    suitability_challenges = []
    
    # Check sample size
    if sample_size >= 100:
        clustering_suitability_score += 0.2
        suitability_reasons.append("sufficient data points")
    else:
        suitability_challenges.append("small sample size")
    
    # Check feature count
    if 2 <= feature_count <= 20:
        clustering_suitability_score += 0.2
        suitability_reasons.append("appropriate feature count")
    elif feature_count > 20:
        suitability_challenges.append("high dimensionality")
    
    # Check for missing values
    missing_ratio = X_sample.isnull().sum().sum() / (len(X_sample) * len(numeric_cols))
    if missing_ratio < 0.05:
        clustering_suitability_score += 0.1
        suitability_reasons.append("minimal missing values")
    elif missing_ratio > 0.2:
        suitability_challenges.append("many missing values")
    
    # Optimal cluster analysis using elbow method
    if sample_size >= 10 and feature_count >= 2:
        try:
            X_scaled = StandardScaler().fit_transform(X_sample)
            
            # Elbow method analysis
            K_range = range(2, min(11, sample_size//10 + 2))
            inertias = []
            silhouette_scores = []
            
            for k in K_range:
                if k < sample_size:
                    kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(float(kmeans.inertia_))
                    if k < sample_size:
                        sil_score = silhouette_score(X_scaled, kmeans.labels_)
                        silhouette_scores.append(float(sil_score))
                    else:
                        silhouette_scores.append(0.0)
            
            # Find optimal k
            optimal_k_silhouette = K_range[np.argmax(silhouette_scores)] if silhouette_scores else 3
            
            optimal_clusters_analysis = {
                "elbow_method": {
                    "k_range": list(K_range),
                    "inertias": inertias,
                    "suggested_range": [2, min(8, sample_size//20 + 2)],
                    "optimal_k": int(optimal_k_silhouette)
                },
                "silhouette_analysis": {
                    "k_range": list(K_range),
                    "silhouette_scores": silhouette_scores,
                    "best_k": int(optimal_k_silhouette),
                    "best_score": float(max(silhouette_scores)) if silhouette_scores else 0.0
                }
            }
        except:
            optimal_clusters_analysis = {
                "elbow_method": {"suggested_range": [2, 5], "optimal_k": 3},
                "silhouette_analysis": {"best_k": 3, "best_score": 0.0}
            }
    else:
        optimal_clusters_analysis = {
            "elbow_method": {"suggested_range": [2, 5], "optimal_k": 3},
            "silhouette_analysis": {"best_k": 3, "best_score": 0.0}
        }
    
    # Algorithm recommendations with detailed analysis
    algorithm_details = {
        "kmeans": {
            "name": "K-Means",
            "confidence": 0.9 if sample_size >= 50 and feature_count <= 20 else 0.7,
            "reasons": ["spherical clusters", "numeric data", "scalable"],
            "optimal_clusters": f"{optimal_clusters_analysis['silhouette_analysis']['best_k']}-{optimal_clusters_analysis['silhouette_analysis']['best_k']+2}",
            "pros": ["fast execution", "interpretable results", "scalable to large datasets"],
            "cons": ["assumes spherical clusters", "sensitive to outliers", "requires cluster count"],
            "best_for": ["well-separated spherical clusters", "large datasets", "known cluster count"],
            "complexity": "low",
            "scalability": "high"
        },
        "hierarchical": {
            "name": "Hierarchical Clustering",
            "confidence": 0.8 if sample_size <= 1000 else 0.5,
            "reasons": ["no cluster count needed", "dendogram visualization", "nested relationships"],
            "optimal_clusters": "determined by dendogram",
            "pros": ["no cluster count needed", "dendogram visualization", "deterministic results"],
            "cons": ["slow on large datasets", "sensitive to noise", "memory intensive"],
            "best_for": ["small to medium datasets", "unknown cluster count", "hierarchical relationships"],
            "complexity": "medium",
            "scalability": "low"
        },
        "dbscan": {
            "name": "DBSCAN",
            "confidence": 0.7 if data_density == "dense" else 0.5,
            "reasons": ["handles irregular shapes", "finds outliers", "no cluster count needed"],
            "optimal_clusters": "automatically determined",
            "pros": ["finds irregular shapes", "handles noise/outliers", "no cluster count needed"],
            "cons": ["sensitive to parameters", "struggles with varying densities", "parameter tuning needed"],
            "best_for": ["irregular cluster shapes", "noisy data", "outlier detection"],
            "complexity": "medium",
            "scalability": "medium"
        },
        "gaussian_mixture": {
            "name": "Gaussian Mixture Model",
            "confidence": 0.8 if feature_count <= 10 else 0.6,
            "reasons": ["probabilistic clustering", "overlapping clusters", "soft assignments"],
            "optimal_clusters": f"{optimal_clusters_analysis['silhouette_analysis']['best_k']}",
            "pros": ["probabilistic assignments", "handles overlapping clusters", "provides uncertainty"],
            "cons": ["assumes gaussian distributions", "sensitive to initialization", "computationally intensive"],
            "best_for": ["overlapping clusters", "probabilistic assignments", "gaussian-distributed data"],
            "complexity": "high",
            "scalability": "medium"
        }
    }
    
    # Determine recommended algorithm
    if sample_size >= 100 and feature_count <= 15:
        recommended_algorithm = "kmeans"
    elif sample_size <= 500:
        recommended_algorithm = "hierarchical"
    elif data_density == "dense":
        recommended_algorithm = "dbscan"
    else:
        recommended_algorithm = "kmeans"
    
    # Feature analysis for clustering
    feature_correlations = {}
    scaling_needed = False
    
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = X_sample.corr()
            # Get high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": float(corr_val)
                        })
            
            # Convert correlation matrix to serializable format
            corr_dict = {}
            for col1 in corr_matrix.columns:
                corr_dict[col1] = {}
                for col2 in corr_matrix.columns:
                    corr_dict[col1][col2] = float(corr_matrix.loc[col1, col2])
            
            feature_correlations = {
                "high_correlations": high_corr_pairs[:5],  # Top 5
                "correlation_matrix": corr_dict
            }
            
            # Check if scaling is needed
            feature_ranges = X_sample.max() - X_sample.min()
            max_range = feature_ranges.max()
            min_range = feature_ranges.min()
            if max_range / min_range > 10:
                scaling_needed = True
                
        except:
            feature_correlations = {"high_correlations": [], "correlation_matrix": {}}
    
    detailed_data = {
        "clustering_analysis": {
            "dataset_characteristics": {
                "sample_size": "large" if sample_size > 1000 else "medium" if sample_size > 100 else "small",
                "sample_count": int(sample_size),
                "feature_count": int(feature_count),
                "numeric_features": int(feature_count),
                "categorical_features": int(categorical_features),
                "data_density": data_density,
                "dimensionality": dimensionality,
                "missing_values_ratio": float(missing_ratio)
            },
            "clustering_suitability": {
                "overall_score": float(min(1.0, clustering_suitability_score)),
                "interpretation": "excellent" if clustering_suitability_score >= 0.8 else "good" if clustering_suitability_score >= 0.6 else "fair",
                "reasons": suitability_reasons,
                "challenges": suitability_challenges,
                "preprocessing_needed": scaling_needed or missing_ratio > 0.05
            }
        },
        "algorithm_recommendations": {
            "recommended": {
                "name": recommended_algorithm,
                "details": algorithm_details[recommended_algorithm]
            },
            "all_algorithms": algorithm_details,
            "total_available": len(algorithm_details)
        },
        "optimal_clusters_analysis": optimal_clusters_analysis,
        "feature_analysis": {
            "clustering_features": numeric_cols,
            "feature_count": int(feature_count),
            "feature_correlations": feature_correlations,
            "scaling_needed": scaling_needed,
            "dimensionality_reduction": "recommended" if feature_count > 10 else "optional"
        },
        "data_preview": df[numeric_cols].head().to_dict('records') if len(numeric_cols) > 0 else []
    }

    # Store detailed data for frontend
    # PIPELINE_STATE["last_detailed_data"] = detailed_data

    # Before storing detailed data, clean it
    detailed_data = convert_numpy_types(detailed_data)

    # Store detailed data for frontend
    PIPELINE_STATE["last_detailed_data"] = detailed_data

    return {
        "algorithms_shown": True,
        "available_algorithms": ["kmeans", "hierarchical", "dbscan", "gaussian_mixture"],
        "numeric_features": len(numeric_cols),
        "recommended_algorithm": recommended_algorithm,
        "formatted_response": response
    }

@tool
def execute_clustering(path: str, algorithm: str = "recommended", n_clusters: Optional[int] = None) -> Dict[str, Any]:
    """
    Executes clustering with specified algorithm.
    User says 'execute clustering kmeans', 'execute clustering recommended', etc.
    """
    df = get_dataset(path)
    if df is None:
        return {"error": "Dataset not found"}
    
    # Prepare data for clustering (only numeric features)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric features for clustering"}
    
    print("dataset loaded")

    X = df[numeric_cols].copy()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())
    
    # Scale features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("reached step 2")

    # Determine optimal number of clusters if not specified
    if not n_clusters and algorithm != "dbscan":
        # Use elbow method to suggest optimal clusters
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(11, len(df)//10 + 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(float(kmeans.inertia_))
            silhouette_scores.append(float(silhouette_score(X_scaled, kmeans.labels_)))
        
        # Find optimal k using silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        n_clusters = optimal_k
    
    print("2.1")

    # # Define available clustering algorithms
    # algorithms_available = {}
    
    # if algorithm == "kmeans" or algorithm is None:
    #     algorithms_available["kmeans"] = KMeans(
    #         n_clusters=n_clusters or 3, 
    #         random_state=Config.RANDOM_STATE,
    #         n_init=10
    #     )
    
    # print("2.2")

    # if algorithm == "hierarchical" or algorithm is None:
    #     algorithms_available["hierarchical"] = AgglomerativeClustering(
    #         n_clusters=n_clusters or 3
    #     )
    
    # if algorithm == "dbscan" or algorithm is None:
    #     algorithms_available["dbscan"] = DBSCAN(eps=0.5, min_samples=5)
    
    # if algorithm == "gaussian_mixture" or algorithm is None:
    #     algorithms_available["gaussian_mixture"] = GaussianMixture(
    #         n_components=n_clusters or 3,
    #         random_state=Config.RANDOM_STATE
    #     )
    
    # print("2.3")

    # # Handle algorithm parameter - convert to lowercase and handle variations
    # if algorithm:
    #     algorithm = algorithm.lower()
    #     if algorithm in ["k-means", "kmeans"]:
    #         algorithm = "kmeans"
    #     elif algorithm in ["k_means"]:
    #         algorithm = "kmeans"
    
    # print("2.4")

    # # # If specific algorithm requested, use only that one
    # # if algorithm and algorithm in ["kmeans", "hierarchical", "dbscan", "gaussian_mixture"]:
    # #     algorithms_to_run = {algorithm: algorithms_available[algorithm]}
    # # else:
    # #     # Run recommended algorithms
    # #     algorithms_to_run = {
    # #         "kmeans": algorithms_available["kmeans"],
    # #         "hierarchical": algorithms_available["hierarchical"]
    # #     }

    # print("2.4")

    # # Debug the algorithm selection process
    # print(f"DEBUG: algorithm parameter = '{algorithm}'")
    # print(f"DEBUG: algorithms_available keys = {list(algorithms_available.keys())}")

    # try:
    #    # If specific algorithm requested, use only that one
    #     if algorithm and algorithm in ["kmeans", "hierarchical", "dbscan", "gaussian_mixture"]:
    #         print(f"DEBUG: Specific algorithm '{algorithm}' requested")

    #         if algorithm in algorithms_available:
    #             print(f"DEBUG: Algorithm '{algorithm}' found in available algorithms")
    #             algorithms_to_run = {algorithm: algorithms_available[algorithm]}
    #         else:
    #             print(f"ERROR: Algorithm '{algorithm}' not found in algorithms_available")
    #             print(f"Available keys: {list(algorithms_available.keys())}")
    #             return {"error": f"Algorithm '{algorithm}' not available"}
            
    #     else:
    #         print("DEBUG: Using default algorithm set (kmeans + hierarchical)")
        
    #         # Check if required default algorithms exist
    #         required_algorithms = ["kmeans", "hierarchical"]
    #         missing_algorithms = [alg for alg in required_algorithms if alg not in algorithms_available]
        
    #         if missing_algorithms:
    #             print(f"ERROR: Missing required default algorithms: {missing_algorithms}")
    #             return {"error": f"Missing required algorithms: {missing_algorithms}"}
        
    #         algorithms_to_run = {
    #             "kmeans": algorithms_available["kmeans"],
    #             "hierarchical": algorithms_available["hierarchical"]
    #         }
    
    #     print(f"DEBUG: algorithms_to_run keys = {list(algorithms_to_run.keys())}")
        
    #     print("reached step 3")

    # except KeyError as e:
    #     print(f"KeyError in algorithm selection: {e}")
    #     return {"error": f"KeyError: {e}"}
    # except Exception as e:
    #     print(f"Unexpected error in algorithm selection: {e}")
    #     return {"error": f"Algorithm selection error: {e}"}

    
    # # print("reached step 3")

    print("reached step 2")

    # ALWAYS build all algorithms with default parameters - no conditional logic
    algorithms_available = {
        "kmeans": KMeans(
            n_clusters=n_clusters or 3, 
            random_state=Config.RANDOM_STATE,
            n_init=10
        ),
        "hierarchical": AgglomerativeClustering(
            n_clusters=n_clusters or 3
        ),
        "dbscan": DBSCAN(eps=0.5, min_samples=5),
        "gaussian_mixture": GaussianMixture(
            n_components=n_clusters or 3,
            random_state=Config.RANDOM_STATE
        )
    }

    print(f"DEBUG: algorithms_available keys = {list(algorithms_available.keys())}")

    # Handle algorithm parameter - normalize input
    if algorithm:
        algorithm = algorithm.lower()
        # Handle common variations
        if algorithm in ["k-means", "k_means"]:
            algorithm = "kmeans"

    print("2.4")

    # Algorithm selection with validation
    if algorithm and algorithm in algorithms_available:
        algorithms_to_run = {algorithm: algorithms_available[algorithm]}
        print(f"DEBUG: Running specific algorithm: {algorithm}")
    elif algorithm:
        # Invalid algorithm requested
        available_list = list(algorithms_available.keys())
        return {"error": f"Algorithm '{algorithm}' not supported. Available: {available_list}"}
    else:
        # No algorithm specified - use recommended defaults
        algorithms_to_run = {
            "kmeans": algorithms_available["kmeans"],
            "hierarchical": algorithms_available["hierarchical"]
        }
        print("DEBUG: Running recommended algorithms: kmeans, hierarchical")

    print(f"DEBUG: algorithms_to_run keys = {list(algorithms_to_run.keys())}")
    print("reached step 3")

    # Perform clustering
    results = {}
    for alg_name, clusterer in algorithms_to_run.items():
        try:
            if alg_name == "gaussian_mixture":
                labels = clusterer.fit_predict(X_scaled)
            else:
                labels = clusterer.fit_predict(X_scaled)
            
            # Calculate metrics
            n_clusters_found = len(np.unique(labels))
            if n_clusters_found > 1:
                silhouette = float(silhouette_score(X_scaled, labels))
                calinski_harabasz = float(calinski_harabasz_score(X_scaled, labels))
            else:
                silhouette = -1.0
                calinski_harabasz = 0.0
            
            # results[alg_name] = {
            #     "labels": labels.tolist() if hasattr(labels, 'tolist') else labels,
            #     "n_clusters": int(n_clusters_found),
            #     "silhouette_score": float(silhouette),
            #     "calinski_harabasz_score": float(calinski_harabasz),
            #     "model": clusterer
            # }
            
            results[alg_name] = {
            "labels": labels.tolist() if hasattr(labels, 'tolist') else [int(x) for x in labels],
            "n_clusters": int(n_clusters_found),
            "silhouette_score": float(silhouette),
            "calinski_harabasz_score": float(calinski_harabasz),
            "model_for_analysis": clusterer  # Keep model only for analysis, don't serialize
            }

        except Exception as e:
            results[alg_name] = {"error": str(e)}
    
    # Find best clustering result
    valid_results = {k: v for k, v in results.items() if "error" not in v and v["silhouette_score"] > 0}
    
    print("reached step 4")
    
    if valid_results:
        best_algorithm = max(valid_results.keys(), key=lambda k: valid_results[k]["silhouette_score"])
        best_labels = valid_results[best_algorithm]["labels"]
        
        # Ensure best_labels is a list of integers
        if hasattr(best_labels, 'tolist'):
            best_labels = best_labels.tolist()
        best_labels = [int(x) for x in best_labels]
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['Cluster'] = best_labels
        DATAFRAME_CACHE[f"{path}_clustered"] = df_clustered
    else:
        return {"error": "All clustering algorithms failed or produced poor results"}
    
    # # Store results
    # PIPELINE_STATE["clustering_results"] = {
    #     "algorithms": valid_results,
    #     "best_algorithm": best_algorithm,
    #     "best_labels": best_labels,
    #     "features_used": numeric_cols,
    #     "n_samples": len(X)
    # }

    print("reached step 5")
    

    # Store results (cleaned for serialization)
    algorithms_summary = {}
    for alg_name, result in valid_results.items():
        if "error" not in result:
            algorithms_summary[alg_name] = {
                "labels": [int(x) for x in result["labels"]] if isinstance(result["labels"], list) else result["labels"],
                "n_clusters": int(result["n_clusters"]),
                "silhouette_score": float(result["silhouette_score"]),
                "calinski_harabasz_score": float(result["calinski_harabasz_score"])
                # Remove "model" key to avoid serialization issues
            }

    PIPELINE_STATE["clustering_results"] = {
        "algorithms": algorithms_summary,
        "best_algorithm": str(best_algorithm),
        "best_labels": [int(x) for x in best_labels],
        "features_used": [str(col) for col in numeric_cols],
        "n_samples": int(len(X))
    }

    
    # Format response
    response = "🎯 **Clustering Analysis Complete**\n\n"
    response += f"**Dataset:** {len(df)} samples, {len(numeric_cols)} features used\n"
    response += f"**Features:** {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}\n\n"
    
    response += "**🏆 CLUSTERING RESULTS:**\n"
    for alg_name, result in valid_results.items():
        status = "👑 BEST" if alg_name == best_algorithm else ""
        response += f"• **{alg_name.replace('_', ' ').title()}:** {result['n_clusters']} clusters, "
        response += f"Silhouette: {result['silhouette_score']:.3f} {status}\n"
    
    best_result = valid_results[best_algorithm]
    response += f"\n**🎯 RECOMMENDED:** {best_algorithm.replace('_', ' ').title()}\n"
    response += f"• **Clusters Found:** {best_result['n_clusters']}\n"
    response += f"• **Silhouette Score:** {best_result['silhouette_score']:.3f}\n"
    response += f"• **Calinski-Harabasz Score:** {best_result['calinski_harabasz_score']:.1f}\n\n"
    
    # Cluster distribution
    unique, counts = np.unique(best_labels, return_counts=True)
    cluster_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    response += "**📊 CLUSTER DISTRIBUTION:**\n"
    for cluster_id, count in cluster_dist.items():
        percentage = float(count / len(best_labels) * 100)
        response += f"• Cluster {cluster_id}: {count} samples ({percentage:.1f}%)\n"
    
    response += "\n**💡 NEXT STEPS:**\n"
    response += "• Type 'analyze clusters' for detailed cluster analysis\n"
    response += "• Type 'visualize clusters' to see cluster plots\n"
    response += "• Clustered data saved with 'Cluster' column\n"
    
    # Collect detailed data for frontend
    import time
    from sklearn.decomposition import PCA
    from sklearn.metrics import davies_bouldin_score
    
    # best_result = valid_results[best_algorithm]
    # best_labels = best_result["labels"]
    # best_model = best_result["model"]
    
    best_result = valid_results[best_algorithm]
    best_labels = best_result["labels"]
    best_model = results[best_algorithm].get("model_for_analysis")  # Get from original results

    # Clustering results summary
    clustering_results = {
        "algorithm_used": str(best_algorithm),
        "algorithm_display_name": str(best_algorithm.replace('_', ' ').title()),
        "n_clusters": int(best_result['n_clusters']),
        "total_samples": int(len(df)),
        "clustering_time": "< 1 second",  # Placeholder
        "convergence": True,  # Most algorithms converge
        "features_used": [str(col) for col in numeric_cols]
    }
    
    # Cluster statistics
    unique_labels, counts = np.unique(best_labels, return_counts=True)
    cluster_sizes = [int(count) for count in counts]
    cluster_percentages = [float(count / len(best_labels) * 100) for count in counts]
    
    cluster_statistics = {
        "cluster_sizes": cluster_sizes,
        "cluster_percentages": cluster_percentages,
        "largest_cluster": int(np.argmax(counts)),
        "smallest_cluster": int(np.argmin(counts)),
        "size_balance": "well-balanced" if (max(counts) / min(counts)) < 2 else "imbalanced",
        "cluster_count": int(len(unique_labels))
    }
    
    # Quality metrics
    try:
        davies_bouldin = davies_bouldin_score(X_scaled, best_labels)
    except:
        davies_bouldin = 0.0
    
    # Interpret silhouette score
    sil_score = best_result['silhouette_score']
    if sil_score >= 0.7:
        quality_interpretation = "excellent clustering quality"
    elif sil_score >= 0.5:
        quality_interpretation = "good clustering quality"
    elif sil_score >= 0.25:
        quality_interpretation = "fair clustering quality"
    else:
        quality_interpretation = "poor clustering quality"
    
    quality_metrics = {
        "silhouette_score": float(sil_score),
        "calinski_harabasz": float(best_result['calinski_harabasz_score']),
        "davies_bouldin": float(davies_bouldin),
        "inertia": float(getattr(best_model, 'inertia_', 0.0)),
        "interpretation": quality_interpretation,
        "quality_level": "excellent" if sil_score >= 0.7 else "good" if sil_score >= 0.5 else "fair" if sil_score >= 0.25 else "poor"
    }
    
    # Cluster characteristics analysis
    cluster_characteristics = {}
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = best_labels
    
    for i, cluster_id in enumerate(unique_labels):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        cluster_numeric = cluster_data[numeric_cols]
        
        # Calculate centroid
        centroid = cluster_numeric.mean().values
        centroid = [float(x) for x in centroid]
        
        # Find distinguishing features
        overall_means = df[numeric_cols].mean()
        cluster_means = cluster_numeric.mean()
        feature_differences = {}
        key_features = {}
        
        for feature in numeric_cols:
            diff = cluster_means[feature] - overall_means[feature]
            feature_differences[feature] = float(diff)
            
            # Categorize feature levels
            std_dev = df[feature].std()
            if abs(diff) > std_dev:
                if diff > 0:
                    key_features[feature] = "high"
                else:
                    key_features[feature] = "low"
            else:
                key_features[feature] = "average"
        
        # Generate cluster description
        high_features = [f for f, level in key_features.items() if level == "high"]
        low_features = [f for f, level in key_features.items() if level == "low"]
        
        if high_features and low_features:
            description = f"High {', '.join(high_features[:2])}, Low {', '.join(low_features[:2])}"
        elif high_features:
            description = f"High {', '.join(high_features[:3])}"
        elif low_features:
            description = f"Low {', '.join(low_features[:3])}"
        else:
            description = "Average across all features"
        
        cluster_characteristics[f"cluster_{cluster_id}"] = {
            "size": int(cluster_sizes[i]),
            "percentage": float(cluster_percentages[i]),
            "centroid": centroid,
            "description": description,
            "key_features": key_features,
            "feature_differences": feature_differences
        }
    
    # Visualization data preparation
    visualization_data = {}
    
    # PCA for 2D visualization if more than 2 features
    if len(numeric_cols) > 2:
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            visualization_data = {
                "pca_data": {
                    "x_values": [float(x) for x in X_pca[:, 0]],
                    "y_values": [float(x) for x in X_pca[:, 1]],
                    "cluster_labels": [int(x) for x in best_labels],
                    "explained_variance": [float(x) for x in pca.explained_variance_ratio_],
                    "total_variance_explained": float(sum(pca.explained_variance_ratio_))
                },
                "feature_importance": {
                    "pc1_features": dict(zip(numeric_cols, [float(x) for x in pca.components_[0]])),
                    "pc2_features": dict(zip(numeric_cols, [float(x) for x in pca.components_[1]]))
                }
            }
        except:
            visualization_data = {"pca_data": None, "feature_importance": {}}
    else:
        # Use original features for 2D plot
        visualization_data = {
            "scatter_data": {
                "x_values": [float(x) for x in X_scaled[:, 0]],
                "y_values": [float(x) for x in X_scaled[:, 1]],
                "cluster_labels": [int(x) for x in best_labels],
                "x_feature": numeric_cols[0],
                "y_feature": numeric_cols[1]
            }
        }
    
    # Cluster centers for visualization
    if hasattr(best_model, 'cluster_centers_'):
        cluster_centers = best_model.cluster_centers_
        if len(numeric_cols) > 2 and 'pca_data' in visualization_data:
            # Transform centers to PCA space
            try:
                centers_pca = pca.transform(cluster_centers)
                visualization_data["cluster_centers"] = {
                    "x_centers": [float(x) for x in centers_pca[:, 0]],
                    "y_centers": [float(x) for x in centers_pca[:, 1]]
                }
            except:
                pass
        elif len(numeric_cols) == 2:
            visualization_data["cluster_centers"] = {
                "x_centers": [float(x) for x in cluster_centers[:, 0]],
                "y_centers": [float(x) for x in cluster_centers[:, 1]]
            }
    
    # Update cache with clustered data
    from helpers import get_cache_key
    cache_key = get_cache_key(path)
    if cache_key:
        DATAFRAME_CACHE[cache_key] = df_with_clusters
    else:
        DATAFRAME_CACHE[path] = df_with_clusters
    
    detailed_data = {
        "clustering_results": clustering_results,
        "cluster_statistics": cluster_statistics,
        "quality_metrics": quality_metrics,
        "cluster_characteristics": cluster_characteristics,
        "visualization_data": visualization_data,
        # "algorithm_comparison": {
        #     alg_name: {
        #         "silhouette_score": float(result['silhouette_score']),
        #         "n_clusters": int(result['n_clusters']),
        #         "calinski_harabasz": float(result['calinski_harabasz_score'])
        #     }
        #     for alg_name, result in valid_results.items()
        # },
        "algorithm_comparison": {
            alg_name: {
            "silhouette_score": float(result['silhouette_score']),
            "n_clusters": int(result['n_clusters']),
            "calinski_harabasz": float(result['calinski_harabasz_score'])
        }
    for alg_name, result in algorithms_summary.items()  # Use cleaned results
},

        "data_summary": {
            "original_features": [str(col) for col in numeric_cols],
            "samples_clustered": int(len(df)),
            "preprocessing_applied": ["standardization", "missing_value_imputation"],
            "cluster_distribution": {str(k): int(v) for k, v in cluster_dist.items()}
        }
    }

    # Store detailed data for frontend
    PIPELINE_STATE["last_detailed_data"] = detailed_data

    return {
        "clustering_complete": True,
        "best_algorithm": best_algorithm,
        "n_clusters": best_result['n_clusters'],
        "silhouette_score": best_result['silhouette_score'],
        "formatted_response": response,
        "cluster_distribution": cluster_dist
    }

@tool
def analyze_clusters(path: str) -> Dict[str, Any]:
    """
    Provides detailed analysis of clustering results.
    User says 'analyze clusters' or 'cluster analysis'.
    """
    if "clustering_results" not in PIPELINE_STATE:
        return {"error": "No clustering results found. Please perform clustering first."}
    
    clustering_info = PIPELINE_STATE["clustering_results"]
    clustered_path = f"{path}_clustered"
    
    if clustered_path not in DATAFRAME_CACHE:
        return {"error": "Clustered dataset not found"}
    
    df_clustered = DATAFRAME_CACHE[clustered_path]
    numeric_cols = clustering_info["features_used"]
    
    # Analyze each cluster
    cluster_analysis = {}
    for cluster_id in df_clustered['Cluster'].unique():
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        
        # Convert pandas series to dict with float values
        means_dict = {}
        stds_dict = {}
        for col in numeric_cols:
            means_dict[col] = float(cluster_data[col].mean())
            stds_dict[col] = float(cluster_data[col].std())
        
        analysis = {
            "size": int(len(cluster_data)),
            "percentage": float(len(cluster_data) / len(df_clustered) * 100),
            "feature_means": means_dict,
            "feature_stds": stds_dict
        }
        
        cluster_analysis[f"Cluster_{cluster_id}"] = analysis
    
    # Format response
    response = "📊 **Detailed Cluster Analysis**\n\n"
    response += f"**Algorithm Used:** {clustering_info['best_algorithm'].replace('_', ' ').title()}\n"
    response += f"**Total Clusters:** {len(cluster_analysis)}\n\n"
    
    for cluster_name, analysis in cluster_analysis.items():
        response += f"**{cluster_name}** ({analysis['size']} samples, {analysis['percentage']:.1f}%)\n"
        
        # Show top distinguishing features
        response += "• **Key Characteristics:**\n"
        means = analysis['feature_means']
        top_features = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        for feature, value in top_features:
            response += f"  - {feature}: {value:.2f}\n"
        response += "\n"
    
    response += "**💡 INSIGHTS:**\n"
    response += "• Compare cluster characteristics to understand patterns\n"
    response += "• Use cluster labels for further analysis or modeling\n"
    response += "• Consider business interpretation of each cluster\n"
    
    return {
        "analysis_complete": True,
        "cluster_count": len(cluster_analysis),
        "formatted_response": response,
        "cluster_details": cluster_analysis
    }