import pandas as pd
from sklearn.model_selection import train_test_split

def split(label_for_training: pd.DataFrame, df_features_for_training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df_features_for_training, label_for_training, test_size=0.2, random_state=42)