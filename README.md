# Sentiment Analysis and Clustering Analysis

This repository contains a collection of Jupyter notebooks for performing sentiment analysis on reviews from multiple platforms and creating visualizations of sentiment patterns across clusters.

## Notebooks Overview

### 1. **sentimentAnalysisAirbnb.ipynb**
Performs comprehensive sentiment analysis on Airbnb reviews using two complementary approaches:
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A lexicon-based sentiment analysis tool that provides quick, interpretable results
- **RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)**: A pre-trained transformer model that provides more nuanced sentiment classification
- Splits reviews into individual sentences for more granular sentiment analysis
- Generates sentiment score distributions and creates output CSV files with results
- Outputs: `results_df_*.csv` with sentiment scores for each review

### 2. **sentimentAnalysisHotels.ipynb**
Performs the same sentiment analysis workflow as the Airbnb notebook but on hotel review data:
- Uses both VADER and RoBERTa sentiment analysis methods
- Analyzes Hotel Reviews dataset
- Produces sentiment scores and visualizations for hotel reviews
- Helps identify sentiment patterns specific to the hospitality sector

### 3. **sentimentAnalysisTripAdvisor.ipynb**
Extends sentiment analysis to TripAdvisor reviews:
- Applies the same dual-model sentiment analysis (VADER + RoBERTa)
- Analyzes travel and tourism reviews from the TripAdvisor platform
- Provides comparative insights across different review platforms
- Generates sentiment distributions and score outputs

### 4. **airbnbDataAnalysisLeaevOneOut.ipynb**
Advanced analysis notebook that combines clustering with sentiment analysis:
- **Data Integration**: Merges pre-clustered Airbnb listing data with sentiment analysis results
- **Cluster Analysis**: Analyzes sentiment patterns within each cluster (0-4)
- **Word Cloud Generation**: Creates distinctive word clouds for each cluster showing:
  - **Positive words** (CxP.png): Words that appear more frequently in positive reviews
  - **Negative words** (CxN.png): Words that appear more frequently in negative reviews
- **Contrastive Analysis**: Compares word frequencies between positive and negative reviews within each cluster
- **Output**: Saves word clouds to `WordClouds15/` folder with naming convention `C0P.png`, `C0N.png`, `C1P.png`, etc.
