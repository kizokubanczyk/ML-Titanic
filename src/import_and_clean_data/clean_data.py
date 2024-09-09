import pandas as pd
import numpy as np

def clean(Titanic_dataFrame_train: pd.DataFrame, Titanic_dataFrame_test: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame ,pd.DataFrame]:
     Titanic_dataFrame_test = Titanic_dataFrame_test.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis = 1)
     Titanic_dataFrame_test['Age'] = Titanic_dataFrame_test['Age'].fillna(Titanic_dataFrame_test['Age'].mean())
     Titanic_dataFrame_test['Fare'] = Titanic_dataFrame_test['Fare'].fillna(Titanic_dataFrame_test['Fare'].mean())

     Sex_map = {'male': 1, 'female': 0}
     Titanic_dataFrame_test['Sex'] = Titanic_dataFrame_test['Sex'].map(Sex_map)

     Embarked_map = {'S': 1, 'C': 2, 'Q': 3}
     Titanic_dataFrame_test['Embarked'] = Titanic_dataFrame_test['Embarked'].map(Embarked_map)


     Titanic_dataFrame_train = Titanic_dataFrame_train.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis = 1)
     Titanic_dataFrame_train['Age'] = Titanic_dataFrame_train['Age'].fillna(Titanic_dataFrame_train['Age'].mean())
     Titanic_dataFrame_train.dropna(subset=['Embarked'], inplace=True)

     Titanic_dataFrame_train['Sex'] = Titanic_dataFrame_train['Sex'].map(Sex_map)
     Titanic_dataFrame_train['Embarked'] = Titanic_dataFrame_train['Embarked'].map(Embarked_map)


     df_label_train= Titanic_dataFrame_train[['Survived']]
     df_features_train = Titanic_dataFrame_train.drop(['Survived', 'SibSp', 'Parch', 'Embarked'], axis = 1)

     df_label_test = Titanic_dataFrame_test[['Survived']]
     df_features_test = Titanic_dataFrame_test.drop(['Survived', 'SibSp', 'Parch', 'Embarked'], axis=1)

     return df_label_train, df_features_train, df_label_test, df_features_test


