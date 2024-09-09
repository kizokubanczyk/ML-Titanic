from import_and_clean_data import import_data, clean_data
from model.classifiers.Decision_Tree import Decision_Tree
from model.classifiers.Random_forest import Random_forest

import click
import yaml


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run(config) -> None:
    with (open(config, 'r') as file):
        config_data = yaml.safe_load(file)

        Titanic_dataFrame_train = import_data.import_data(config_data.get('path_dataSet_for_train'))
        Titanic_dataFrame_test = import_data.import_data(config_data.get('path_dataSet_for_test'))

        label_for_train, df_features_for_train, label_for_test, df_features_for_test = clean_data.clean(Titanic_dataFrame_train, Titanic_dataFrame_test)

        decision_Tree = Decision_Tree(label_for_train, df_features_for_train, label_for_test, df_features_for_test)
        decision_Tree.train_model()
        decision_Tree.predict()
        decision_Tree.model_performance()
        decision_Tree.mis_classified_samples()

        randomForest = Random_forest(label_for_train, df_features_for_train, label_for_test, df_features_for_test)
        randomForest.train_model()
        randomForest.predict()
        randomForest.model_performance()

if __name__ == "__main__":
    run()
