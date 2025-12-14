# Second Chances: Quantifying DBs' Post-Throw Recovery After Deceptive Routes

This repository supports my NFL Big Data Bowl 2026 submission, **Second Chances**.

## Prerequisites

Before running the project locally, make sure you have the following tools and libraries installed:

- **uv**: This project uses `uv` to manage project dependencies. Follow the instructions in the [official uv documentation](https://docs.astral.sh/uv/getting-started/) to install uv if it's not already on your system.

- **Python**: Make sure you have a compatible version of Python installed. Refer to the `pyproject.toml` file for the exact version required.

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository** to your desired folder:

    ```bash
    git clone git@github.com:miguelmendesduarte/big-data-bowl-2026-analytics.git <desired-folder-name>
    ```

2. Navigate to the project folder:

    ```bash
    cd <desired-folder-name>
    ```

3. **Install** the dependencies:

    ```bash
    uv sync --all-extras
    ```

## Usage

After setting up the project, follow these steps to begin analyzing the DBs' post-throw recovery after deceptive routes:

1. Add the **NFL Big Data Bowl 2026 data**:

    Download the official [Big Data Bowl 2026 dataset](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data), and place it into the `data/raw` directory. This data is essential for running the analysis and performing the necessary computations.

2. **Run the cleaning pipeline**:

    The next step is to clean the raw data by running the cleaning pipeline. This will process the raw data and output several cleaned datasets in the `data/cleaned` directory.

    ```bash
    uv run -m src.data_processing.cleaning.clean_data
    ```

3. **Create training and testing datasets**:

    After cleaning the data, the next step is to create the datasets used for training and testing the model that will compute the non-completion probability pre-throw. To do this, run the following command:

    ```bash
    uv run -m src.data_processing.training.create_datasets
    ```

    This will generate two CSV files, `train.csv` and `test.csv`, which will be placed in the `data/processed/training` directory. These datasets are used to train and test the model that **predicts non-completion probabilities before the throw**.

4. **Create inference dataset**:

    After creating the training and testing datasets, you need to generate the dataset that will be used for model inference. This dataset will be used to **make predictions based on new data**. To create it, run:

    ```bash
    uv run -m src.data_processing.inference.create_dataset
    ```

    This will generate an `inference.csv` file, which will be stored in the `data/processed/inference/` directory.

5. **Train the model**:

    The model used in this project is an XGBoost model, which you can train using the following steps.

    - **Modify hyperparameters**: Before training, you can customize the hyperparameters in the `src/core/settings.py` module under the `XGB_PARAM_GRID` variable. This defines a grid of possible values for the hyperparameters such as `n_estimators`, `learning_rate`, and others. If needed, update them based on your experiment.

    - **Run the training pipeline**: To start the training process, use the following command:

        ```bash
        uv run -m src.training.train
        ```

    This will train the XGBoost model using the training dataset (`train.csv`) and validate it against the test dataset (`test.csv`). The results will be logged to **MLflow**.

6. **Monitor the training process with MLflow**:

    MLflow is used for logging experiments. If you'd like to visualize the training process, you can run the **MLflow UI** in a separate terminal window:

    ```bash
    mlflow ui
    ```

    - Once the UI is running, navigate to `http://127.0.0.1:5000` in your browser.

    - Look for the model with the **lowest LogLoss** (for probability predictions) and review the AUC and Brier scores. Additionally, check the **Feature Importance** and **Calibration Curve** plots that are logged as artifacts for each model.

7. **Select the best model and update `MODEL_PATH`**:

    After reviewing the models in the MLflow UI, choose the one with the best results (lowest LogLoss, along with favorable AUC and Brier scores). Once you've identified the best model, take note of the **model ID**.

    - Update the `MODEL_PATH` in the `src/core/settings.py` file with the **model ID** from MLflow. This will point to the trained model and be used in the inference pipeline.

        ```bash
        MODEL_PATH: Path = (
            BASE_DIR
            / "mlruns"
            / "1"
            / "models"
            / "<best_model_id>"
            / "artifacts"
        )
        ```

    Replace `<best_model_id>` with the model ID from MLflow.

8. **Run the inference pipeline**:

    With the best model selected, the next step is to run the inference pipeline. This will generate predictions using the trained model on the inference dataset (`inference.csv`) created earlier. To do this, run:

    ```bash
    uv run -m src.inference
    ```

    This will produce a CSV file with the results, which will be stored in the `data/processed/inference/results.csv` file. This CSV will contain the predicted probabilities based on the inference data.

9. **Compute the Deception and Recovery Scores**:

    Once you have the inference results, you can compute the Deception Score and Recovery Score for each play. These scores are key metrics for analyzing DB performance after deceptive routes.

    To compute these scores, run the following command:

    ```bash
    uv run -m src.metrics.compute_scores
    ```

    This will add a CSV file with the scores for each play, as well as the defender and receiver IDs, in `data/scores.py`.

    You can then use these scores to:

    - Analyze DB and receiver performance.

    - Test hypotheses on how deception affects recovery.

    - Create visualizations and plots for further insights.

## Questions or Clarifications?

I'm available for any questions or clarifications. If you need help with the setup, running the analysis, or understanding the results, feel free to reach out! 😊
