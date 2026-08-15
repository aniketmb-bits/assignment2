Assignment 2

**Problem Statement**
Implement the following classification models using the dataset chosen above. All
the 6 ML models have to be implemented on the same dataset.
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest

For each of the models above, calculate the following evaluation metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeﬃcient (MCC Score)

# Dataset Description
The dataset provided predictive feature like education , employment status , marital status to predict if the salary is greater than $50K

It can be used to practice machine learning problem like classification. The total number of features in the dataset are 14. 

It is an imbalanced dataset with almost 75% of the samples as negative samples, while only 25% representing postive.

# GitHub Repository link

# Models used


| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| logistic regression l1 regularization | 0.8  | 0.9 | 0.55 | 0.85 | 0.66  | 0.56  |
| logistic regression l2 regularization | 0.8  | 0.9 | 0.55 | 0.84 | 0.66 | 0.56 |
| Decision tree | 0.81 | 0.75 | 0.61 | 0.63 | 0.63 | 0.49 |
| KNN | 0.86 | 0.84 | 0.63 | 0.60 | 0.62 | 0.49 |
| Naive Bayes classifier | 0.55 | 0.76 | 0.36 | 0.95 | 0.51 | 0.34 |
| Random Forest (Ensemble) | 0.86 | 0.92 | 0.83 | 0.56 | 0.67 | 0.60 |

# Observations on chosen dataset
As the dataset is highly imbalanced, accuracy metric is not reliable for the given model. The main comparison metrics would be F1-score, MCC and AUC. Among precision and recall, precision is considered because less false positives are more important in the given data as that would help design the policy based on the given features to assess income band.

| ML Model Name | Observation | 
| :--- | :---: |
| logistic regression l1 regularization |  MCC score is 0.56 indicating moderate correlation between prediction. AUC 0.9 indicates strong prediction power of the model. Precision and F1 score for this model are 0.55 and 0.66 that indicate moderate performance of the model |
| logistic regression l2 regularization | The logistic regression with l2 regularization reports score very similar to l1 regularization, hence there is not much gain obtained. This indicates that the features turned off during l1 regularization are penalized more in l2 regularization case.
| Decision tree | Decision tee model perform better in precision but have low AUC, F1-score and MCC. Lower AUC and MCC indicate that the model's ability to classify income based on given features is worse than logistic regression. The decision tree pruning methods may need to be employed to improve performance. |
| KNN | K-NN model has AUC 0.84, moderate precision, F1 score and MCC. Better AUC indicating good classification performance of the model. MCC indicate moderate positive correlation. |
| Naive Bayes classifier | Naive Bayes classifier has poor AUC and MCC indicating worse performing model.It also reports low precision.|
| Random Forest (Ensemble) | Random Forest algorithm has overall better metrics in comparison to other models above. Thus, Random Forest classifier is a best performing model on the given dataset with parameters. |
