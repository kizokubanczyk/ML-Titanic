import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

class Support_Vector_Machine:
    def __init__(self, df_labels_train: pd.Series, df_features_train: pd.DataFrame, df_labels_test: pd.Series,
                 df_features_test: pd.DataFrame):
        self.__RandomForest = SVC(random_state=42)

        self.__df_labels_train = df_labels_train
        self.__df_features_train = df_features_train

        self.__df_labels_test = df_labels_test
        self.__df_features_test = df_features_test

        self.__labels_predict = None

    def train_model(self) -> None:
        with open("../config.yaml", 'r') as file:
            config_data = yaml.safe_load(file)
            param_grid = config_data.get('support_vector_machine_parameters')

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
                path_model_support_vector_machine_score = config_data.get('path_model_support_vector_machine_score')
                path_model_support_vector_machine_plot = config_data.get('path_model_support_vector_machine_plot')

            with open(path_model_support_vector_machine_score, 'w') as file:
                file.write("Accuracy score: " + str(accuracy) + "\n")
                file.write("Precision score: " + str(precision) + "\n")
                file.write("Recall score: " + str(recall) + "\n")
                file.write("F1 score: " + str(f1) + "\n")
                file.write("Confusion matrix: " + "\n" + str(cm) + "\n")

        else:
            raise ValueError(
                "No predictions have been made. Please call the 'predict' method before calculating performance.")
