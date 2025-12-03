# Sentiment Analysis and Clustering Overview

This repo contains notebooks that run sentiment analysis on Airbnb, hotel, and TripAdvisor reviews, and then link those results to Airbnb listing clusters to see what people like or complain about.

### sentimentAnalysisAirbnb.ipynb
Runs VADER and RoBERTa on Airbnb reviews, breaking them into sentences and exporting sentiment scores.

### sentimentAnalysisHotels.ipynb
Applies the same sentiment pipeline to hotel reviews so we can benchmark and compare model performance.

### sentimentAnalysisTripAdvisor.ipynb
Runs the same sentiment process on TripAdvisor reviews to test model consistency across platforms.

### airbnbDataAnalysisLeaveOneOut.ipynb
Combines Airbnb cluster labels with the sentiment results and generates positive/negative word clouds for each cluster.
