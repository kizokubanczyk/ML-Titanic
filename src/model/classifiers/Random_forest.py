import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
import yaml
import matplotlib.pyplot as plt

class Random_forest:
    def __init__(self, df_labels_train : pd.Series, df_features_train: pd.DataFrame, df_labels_test : pd.Series, df_features_test: pd.DataFrame):
        self.__RandomForest = RandomForestClassifier(random_state=42)

        self.__df_labels_train = df_labels_train
        self.__df_features_train = df_features_train

        self.__df_labels_test = df_labels_test
        self.__df_features_test = df_features_test

        self.__labels_predict = None

    def train_model(self) -> None:
        with open("../config.yaml", 'r') as file:
            config_data = yaml.safe_load(file)
            param_grid = config_data.get('random_forest_parameters')

            self.__grid_search = GridSearchCV(estimator=self.__RandomForest, param_grid=param_grid, cv=5,scoring='accuracy')
            self.__grid_search.fit(self.__df_features_train, self.__df_labels_train)

    def predict(self) -> None:
        self.__labels_predict = self.__grid_search.best_estimator_.predict(self.__df_features_test)

    def model_performance(self) -> None:
        if self.__labels_predict is not None:
            accuracy = accuracy_score(self.__df_labels_test, self.__labels_predict)
            precision = precision_score(self.__df_labels_test, self.__labels_predict, average='weighted')
            recall = recall_score(self.__df_labels_test, self.__labels_predict, average='weighted')
            f1 = f1_score(self.__df_labels_test, self.__labels_predict, average='weighted')
            cm = confusion_matrix(self.__df_labels_test, self.__labels_predict)

            with open("../config.yaml", 'r') as file:
                config_data = yaml.safe_load(file)
                path_random_forest_scores = config_data.get('path_model_Performance_random_fores_score')
                path_random_forest_plot = config_data.get('path_model_Performance_random_forest_plot')

            with open(path_random_forest_scores, 'w') as file:
                file.write("Accuracy score: " + str(accuracy) + "\n")
                file.write("Precision score: " + str(precision) + "\n")
                file.write("Recall score: " + str(recall) + "\n")
                file.write("F1 score: " + str(f1) + "\n")
                file.write("Confusion matrix: " + "\n" + str(cm) + "\n")

            best_rf = self.__grid_search.best_estimator_
            max_trees_to_save = 10

            for num_trees, tree in enumerate(best_rf.estimators_):
                if num_trees >= max_trees_to_save:
                    break
                plt.figure(figsize=(20, 10))
                plot_tree(
                    tree,
                    feature_names=self.__df_features_train.columns,
                    class_names=['Survived', 'Did not survive'],
                    filled=True,
                    rounded=True,
                    fontsize=10
                )
                plt.savefig(f"{path_random_forest_plot}{num_trees}.png")
                plt.close()

        else:
            raise ValueError(
                "No predictions have been made. Please call the 'predict' method before calculating performance.")

