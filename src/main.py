from import_and_clean_data import import_data, clean_data
from model import split_dataFrame
import click
import yaml


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run(config) -> None:
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)

        Titanic_datFrame = import_data.import_data(config_data.get('path_dataSet'))
        label_for_training, df_features_for_training =  clean_data.clean(Titanic_datFrame)

        x_train, x_test, y_train, y_test = split_dataFrame.split(label_for_training, df_features_for_training)



if __name__ == "__main__":
    run()