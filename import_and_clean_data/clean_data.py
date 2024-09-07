import pandas as pd

def clean(Titanic_dataFrame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    Titanic_dataFrame = Titanic_dataFrame.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis = 1)

    Titanic_dataFrame['Age'] = Titanic_dataFrame['Age'].fillna(Titanic_dataFrame['Age'].mean())
    Titanic_dataFrame.dropna(subset=['Embarked'], inplace=True)

    # musze przekonwertować cehcy
    Sex_map = {'male': 1, 'female': 0}
    Titanic_dataFrame['Sex'] = Titanic_dataFrame['Sex'].map(Sex_map)

    Embarked_map = {'S': 1, 'C': 2, 'Q' : 3}
    Titanic_dataFrame['Embarked'] = Titanic_dataFrame['Embarked'].map(Embarked_map)

    label_for_training = Titanic_dataFrame['Survived']
    df_features_for_training = Titanic_dataFrame.drop(['Survived'], axis = 1)

    return label_for_training, df_features_for_training

