# DataScienceProject
🎬 Movie Recommender System: Neural Networks vs Collaborative Filtering
A comprehensive comparison of Deep Neural Networks and Traditional Collaborative Filtering for movie recommendations using the MovieLens 100K dataset.
📊 Project Overview
This project implements and compares three recommendation approaches:

Collaborative Filtering (User-Based with Cosine Similarity)
Neural Collaborative Filtering (Deep Learning with Embeddings)
Hybrid Model (Weighted Ensemble)

Key Results
ModelRMSEMAETraining TimeCollaborative Filtering0.9230.7414.04sNeural Network0.7820.61452.31sHybrid Model0.7100.578~56s
Achievements:

✅ 15.2% improvement with Neural Networks over CF
✅ 23.1% improvement with Hybrid Model
✅ 50% better cold start performance for new users

🚀 Quick Start
Run in Google Colab (Recommended)
python# Copy the complete code from movie_recommender.py
# Upload to Google Colab and run all cells
Run Locally
bash# Clone repository
git clone https://github.com/Siphephelo-dev/DataScienceProject.git


# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

# Run the script
python movie_recommender.py
📁 Dataset
MovieLens 100K - Automatically downloaded by the script

100,000 ratings from 943 users on 1,682 movies
Rating scale: 1-5 stars
Sparsity: 93.7%
Source: GroupLens Research

🔬 Methodology
1. Data Preprocessing

Load ratings and movie data
Create user/movie index mappings
Split into train (80%) and test (20%) sets

2. Exploratory Data Analysis
Generates 6 visualizations analyzing:

Rating distribution
User activity patterns
Movie popularity
Average ratings
Correlation heatmaps

3. Model Training
Collaborative Filtering:
pythoncf_model = CollaborativeFiltering(n_neighbors=50)
cf_model.fit(train_data, n_users=943, n_movies=1682)
Neural Collaborative Filtering:
Architecture:
- User Embedding (943 × 50)
- Movie Embedding (1,682 × 50)
- Dense(128) + BatchNorm + Dropout(0.3)
- Dense(64) + BatchNorm + Dropout(0.3)
- Dense(32) + ReLU
- Output(1)

Total Parameters: 155,315
Hybrid Model:
pythonhybrid_model = HybridRecommender(cf_model, nn_model, alpha=0.4)
# Prediction = 0.4 × CF + 0.6 × NN
📈 Results
Performance Metrics

RMSE: Hybrid achieves 0.710 (23.1% improvement over CF)
MAE: Hybrid achieves 0.578 (22.0% improvement over CF)

Cold Start Analysis
User TypeCF RMSENN RMSEHybrid RMSENew Users (<5 ratings)1.450.890.72Active Users (>50 ratings)0.760.720.69
📊 Visualizations
The script generates:

eda_analysis.png - 6-panel exploratory data analysis
training_history.png - Neural network loss/MAE curves
model_comparison.png - Performance comparison charts

🔑 Key Findings

Neural networks significantly outperform traditional CF (15.2% better RMSE)
Hybrid approach achieves best results by combining strengths of both methods
Cold start problem effectively addressed - 50% improvement for new users
Scalability: Neural networks scale better with GPU acceleration for large datasets

📦 Dependencies
txtnumpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
tensorflow>=2.10.0
