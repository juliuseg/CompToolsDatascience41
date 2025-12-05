# Sentiment Analysis and Clustering Overview
### ComprehensiveSentimentAnalysis.ipynb

This notebook includes datacleaning of the reviews and sentiment analysis for Airbnb, hotel, and TripAdvisor reviews. Section 3 has results of logistical and SVM performance. 4.5 has performance of VADER and RoBERTa. The wordclouds used in the report can be found in section 6. 

Airbnb data should be downloaded and put in /data. Too large for github. It can be found under reviews for:
AirBNB: https://insideairbnb.com/copenhagen/



### DataCleaning_Clustering.ipynb

The notebook contains the data cleaning and the clustering of listings.csv. The first 2 sections are part of our final solution. The second 2 sections contain an effort to improve the clustering, but we decided not to use them in our final solution.

In the second part, there are 2 blocks where the code calls Gemini for a specific task, which is time consuming and requires an API key to run. We uploaded the results of these blocks as separate files which are read by the code later, and these special parts are commented out.




<!-- 
This repo contains notebooks that run sentiment analysis on Airbnb, hotel, and TripAdvisor reviews, and then link those results to Airbnb listing clusters to see what people like or complain about.

### sentimentAnalysisAirbnb.ipynb
Runs VADER and RoBERTa on Airbnb reviews, breaking them into sentences and exporting sentiment scores.

### sentimentAnalysisHotels.ipynb
Applies the same sentiment pipeline to hotel reviews so we can benchmark and compare model performance.

### sentimentAnalysisTripAdvisor.ipynb
Runs the same sentiment process on TripAdvisor reviews to test model consistency across platforms.

### airbnbDataAnalysisLeaveOneOut.ipynb
Combines Airbnb cluster labels with the sentiment results and generates positive/negative word clouds for each cluster. -->
