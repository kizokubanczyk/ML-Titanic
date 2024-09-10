import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import yaml
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from pandas.plotting import table


class Decision_Tree:
    def __init__(self, df_labels_train:  pd.Series, df_features_train: pd.DataFrame, df_labels_test:  pd.Series, df_features_test: pd.DataFrame):
        self.__decision_tree = DecisionTreeClassifier(random_state=42)

        self.__df_labels_train = df_labels_train
        self.__df_features_train = df_features_train

        self.__df_labels_test = df_labels_test
        self.__df_features_test = df_features_test

        self.__labels_predict = None

    def train_model(self) -> None:

        with open("../config.yaml", 'r') as file:
            config_data = yaml.safe_load(file)
            param_grid = config_data.get('decision_tree_parameters')

        self.__grid_search = GridSearchCV(estimator=self.__decision_tree, param_grid=param_grid, cv=5, scoring='accuracy')

        self.__grid_search.fit(self.__df_features_train, self.__df_labels_train)

    def predict(self) -> None:
        self.__labels_predict =  self.__grid_search.best_estimator_.predict(self.__df_features_test)

    def model_performance(self) -> None:
        if self.__labels_predict is not None:
            accuracy = accuracy_score(self.__df_labels_test, self.__labels_predict)
            precision = precision_score(self.__df_labels_test, self.__labels_predict, average='weighted')
            recall = recall_score(self.__df_labels_test, self.__labels_predict, average='weighted')
            f1 = f1_score(self.__df_labels_test, self.__labels_predict, average='weighted')
            cm = confusion_matrix(self.__df_labels_test, self.__labels_predict)

            with open("../config.yaml", 'r') as file:
                config_data = yaml.safe_load(file)
                path_decision_tree_scores = config_data.get('path_model_Performance_decision_tree_score')
                path_decision_tree_plot = config_data.get('path_model_Performance_decision_tree_plot')

            with open(path_decision_tree_scores, 'w') as file:
                file.write("Accuracy score: " + str(accuracy) + "\n")
                file.write("Precision score: " + str(precision) + "\n")
                file.write("Recall score: " + str(recall) + "\n")
                file.write("F1 score: " + str(f1) + "\n")
                file.write("Confusion matrix: " + "\n" + str(cm) + "\n")

            plt.figure(figsize=(20, 10))
            plot_tree(
                self.__grid_search.best_estimator_,
                feature_names=self.__df_features_train.columns,
                class_names=['Survived', 'Did not survive'],
                filled=True,
                rounded=True,
                fontsize=10
            )
            plt.savefig(path_decision_tree_plot)
            plt.close()

        else:
            raise ValueError(
                "No predictions have been made. Please call the 'predict' method before calculating performance.")

    def mis_classified_samples(self) -> None:

        results = pd.DataFrame({
            'Actual': self.__df_labels_test,
            'Predicted': self.__labels_predict
        })
        show_misclassified_samples = results[results['Actual'] != results['Predicted']]
        mis_classified_indices = show_misclassified_samples.index
        samples = self.__df_features_test.loc[mis_classified_indices]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        table(ax, samples, loc='center', cellLoc='center')

        with open("../config.yaml", 'r') as file:
            config_data = yaml.safe_load(file)
            plt.savefig(config_data.get('path_mis_classified_samples'), bbox_inches='tight', pad_inches=0.05)
            plt.close()


