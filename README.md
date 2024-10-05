# Machine Learning Titanic Project

The aim of this project is to analyse the Titanic dataset and create models that predict which people are more likely to survival. It is a binary classification where the name of the target column is "Survived" indicates whether a person survived 1 or did not survive 0.

I used Jupyter Notebook to analyze the data and select optimal features for the models. I implemented three basic models: Support Vector Machine, Decision Tree, and Random Forest.

Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset/data
## Definitions of three basic models:

- **Support Vector Machine (SVM):**
  This model is a powerful classification algorithm that aims to find a hyperplane in multidimensional space that maximizes the margin between different classes.

- **DecisionTreeClassifier:**
  This model is a classification algorithm that creates a tree-like model where each internal node represents a feature or attribute.

- **RandomForestClassifier:**
  This model is an ensemble classification algorithm that utilizes multiple decision trees during training. Each decision tree is trained on different subsets of data and random subsets of features. The final class (for classification) is determined by voting or averaging the predictions of the individual trees, which enhances accuracy and stability.

  ## Used Evaluation Metrics
- **Accuracy:** The ratio of correctly classified samples to the total number of samples, indicating the model's overall effectiveness.
- **Precision:** Measures the accuracy in identifying positive examples, calculated as true positives divided by the total of true positives and false positives.
- **Recall:** Indicates how well the model identifies all positive samples, computed as true positives divided by the total of true positives and false negatives.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics, especially in imbalanced datasets.

## Model Ranking Based on Performance

1. Support Vector Machine 
2. Decision Tree Classifier
3. RandomF Forest Classifier

To check detailed results, please navigate to the 'scores' folder.

## Decision tree classifier diagram: 

![Screenshot](https://github.com/krystianbrynski/KrystianrBry_ML_Titanic/blob/main/Project_Titanic/scores/Decision_Tree_scores/decision_tree_plot.png)

## Random forest classifier diagrams: 

![Screenshot](https://github.com/krystianbrynski/KrystianrBry_ML_Titanic/blob/main/Project_Titanic/scores/Random_Forest_scores/decision_tree_1.png)
![Screenshot](https://github.com/krystianbrynski/KrystianrBry_ML_Titanic/blob/main/Project_Titanic/scores/Random_Forest_scores/decision_tree_2.png)
