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
