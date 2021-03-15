# ECE 271B Group3: Shopping Recommendation

## By Haoming Zhang and Guoren Zhong

Recommendation system has been a popular topic in the past few decades due to the increasing amount of information in our life. It is an important approach in solving the information overloading problem, and has been applied to many areas, such as product recommenders for online stores and content recommenders for social media platforms. In this work, two algorithms are introduced to solve the Amazon fine food recommendation problem, which are collaborative filtering and latent factor model. These algorithms are used to predict the ratings of users to products, and generate a recommendation list for users. The results of both models are shown in the end.

## Models
### Collaborative filter (CF)
Two types of CF,
+ User-based CF,
+ Item-based CF.
With three similarity rules,
+ Jaccard similarity,
+ Cosine similarity,
+ Pearson correlation coefficient.

### Latent factor model
Gradient descent.

## Implementations

### Data precessing and analysis
+ **notebook/Data_analysis.ipynb**: visualizing data,
+ **notebook/Data_preprocessing.ipynb**: preprocess data (delete columns and rows).

### Collaborative filter
+ **src/DataLoader.py**: load the clean data,
+ **src/Predictions.py**: Make prediction,
+ **src/Similarity.py**: similarity rules used in this work,
+ **src/main.py**: main codes for CF.

### Latent factor model
+ **notebook/Latent_factor_model.ipynb**: Perform latent factor model with gradient descent.

## Data

From Kaggle: "Amazon Fine Food Reviews",

https://www.kaggle.com/snap/amazon-fine-food-reviews.

## Results

For the final recommendations of each user please forward to /ECE271B_Group3/results/recommendations.csv

For the final report of our project please forward to /ECE271B_Group3/results/ECE271B_report_cvpr.pdf
