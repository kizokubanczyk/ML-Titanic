import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class Decision_Tree:
    def __init__(self, _df_labels_train : pd.Series, _df_features_train: pd.DataFrame, _df_labels_test : pd.Series, _df_features_test: pd.DataFrame):
        self.__decision_tree = DecisionTreeClassifier()

        self.__df_labels_train = _df_labels_train
        self.__df_features_train = _df_features_train

        self.__df_labels_test = _df_labels_test
        self.__df_features_test = _df_features_test

        self.__labels_predict = None

    def train_model(self) -> None:
        self.__decision_tree.fit(self.__df_features_train, self.__df_labels_train)

    def predict(self) -> None:
        self.__labels_predict = self.__decision_tree.predict(self.__df_features_test)

    def accuracy(self) -> float:
        if self.__labels_predict is not None:
            accuracy = accuracy_score(self.__df_labels_test, self.__labels_predict)
            return accuracy
        else:
            return 0;





