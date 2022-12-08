from sklearn.model_selection import train_test_split

from game_rater import pipeline
from game_rater.configs.config import config
from game_rater.utils.data_utils import load_dataset, save_pipeline


# this again is repeating the sample project, since the process is the same
def run_training() -> None:
    """Train the model."""
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.model_config.target, axis=1),
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    pipeline.pipe.fit(X_train, y_train)

    save_pipeline(pipeline_to_persist=pipeline.pipe)


if __name__ == "__main__":
    run_training()
