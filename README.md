# Machine Learning Project: Student Placement Prediction

This repository contains a machine learning project focused on predicting student placement status based on various academic and personal attributes. The project demonstrates a complete machine learning pipeline, including data preprocessing, training multiple classification models, hyperparameter tuning, model evaluation, and implementing an ensemble (Voting Classifier).

## Project Overview

The goal of this project is to build a predictive model that can accurately determine whether a student will be "Placed" or "Not Placed" after their academic program. This is a binary classification problem.

## Dataset

The dataset used for this project is `train.csv`. It contains information about students, including:

* `gender`: Gender of the student

* `ssc_p`: Secondary School Certificate (10th Grade) percentage

* `ssc_b`: Board of Secondary School Certificate (Central/Others)

* `hsc_p`: Higher Secondary Certificate (12th Grade) percentage

* `hsc_b`: Board of Higher Secondary Certificate (Central/Others)

* `hsc_s`: Specialisation in Higher Secondary Education (Commerce/Science/Arts)

* `degree_p`: Degree percentage

* `degree_t`: Type of Degree (Comm&Mgmt/Sci&Tech/Others)

* `workex`: Work Experience (Yes/No)

* `etest_p`: Employability Test percentage

* `specialisation`: MBA specialisation (Mkt&HR/Mkt&Fin)

* `mba_p`: MBA percentage

* `status`: Placement status (Placed/Not Placed) - **This is our target variable.**

* `salary`: Salary offered (only for placed students) - *This feature is dropped during preprocessing to avoid target leakage for the classification task.*

## Data Preprocessing

The raw data undergoes several preprocessing steps to prepare it for model training:

1.  **Column Dropping**: The `sl_no` (serial number) and `salary` columns are removed. `salary` is dropped to prevent target leakage, as it directly indicates placement status and would provide an unfair advantage to the model.

2.  **Missing Value Handling**: The dataset was inspected for missing values. Fortunately, after dropping the `salary` column, no missing values were found in the remaining features relevant for the classification task.

3.  **Target Variable Encoding**: The categorical target variable `status` ('Placed', 'Not Placed') is converted into numerical representation (1 and 0 respectively) using `LabelEncoder`.

4.  **Feature Categorization**: Features are identified as either numerical or categorical based on their data types.

5.  **Data Transformation (`ColumnTransformer`)**:

    * **Numerical Features**: `StandardScaler` is applied to numerical features to standardize their scale, which helps many machine learning algorithms perform better.

    * **Categorical Features**: `OneHotEncoder` is used to convert multi-category nominal features into a numerical format, creating new binary columns for each category. This prevents the model from assuming an ordinal relationship between categories.

6.  **Data Splitting**: The preprocessed dataset is split into training and testing sets. A 70% training and 30% testing split is used, with stratification on the `status` column to ensure that the proportions of 'Placed' and 'Not Placed' students are maintained in both sets.

## Models Selected and Training

Four different machine learning classification models were chosen and trained:

1.  **Logistic Regression**: A linear model used for binary classification, serving as a robust baseline.

2.  **Decision Tree Classifier**: A non-linear model that can capture complex decision boundaries and is relatively interpretable.

3.  **Random Forest Classifier**: An ensemble method that builds multiple decision trees and aggregates their predictions, typically offering higher accuracy and robustness than a single decision tree.

4.  **Gradient Boosting Classifier**: Another powerful ensemble technique that builds trees sequentially, with each new tree correcting errors made by previous ones, often leading to excellent performance.

Each model was trained on the preprocessed training data.

### Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used for hyperparameter tuning for each model. The F1-score was selected as the scoring metric for optimization, as it provides a balanced measure of precision and recall, which is crucial for classification tasks where both false positives and false negatives are important.

## Model Evaluation

The performance of each trained model and the Voting Classifier was evaluated on the unseen test set using the following metrics:

* **Accuracy**: The proportion of correctly classified instances.

* **Precision**: The ratio of true positives to all positive predictions. Useful when the cost of false positives is high.

* **Recall (Sensitivity)**: The ratio of true positives to all actual positives. Useful when the cost of false negatives is high.

* **F1-Score**: The harmonic mean of precision and recall, providing a single metric that balances both.

* **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)**: Measures the model's ability to distinguish between classes across various classification thresholds. A higher AUC indicates better discriminatory power.

### Visualizations

* **Confusion Matrices**: Provided for each model and the Voting Classifier to visualize the number of true positives, true negatives, false positives, and false negatives.

* **ROC Curves**: Plotted for all models to graphically represent the trade-off between the True Positive Rate and False Positive Rate at different thresholds.

## Voting Classifier

A `VotingClassifier` was implemented using the best-tuned versions of the four individual models. Soft voting was employed, meaning the final prediction is based on the average of the predicted probabilities from each base model. This ensemble approach often leads to more robust and accurate predictions by leveraging the diverse strengths of individual classifiers.

## Results and Conclusion

A comprehensive table summarizing the performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) for all individual models and the Voting Classifier is provided in the project's output.

Based on the F1-Score, the **[Insert Best Model Name Here, e.g., 'Voting Classifier' or 'Random Forest']** performed the best, demonstrating the most balanced performance in predicting student placement. The Voting Classifier often showcases the benefit of ensemble learning by combining the strengths of multiple models to achieve superior or more stable results than any single model.

## How to Run the Code

To run this project:

### Prerequisites

* Python 3.x

* Required Python libraries (install using pip):

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

### Setup

1.  Clone this repository to your local machine.

2.  Ensure you have the `train.csv` dataset in the same directory as the Python script.

### Usage

1.  Navigate to the directory containing the script in your terminal.

2.  Run the Python script:

    ```bash
    python your_script_name.py # Replace 'your_script_name.py' with the actual filename
    ```

    The script will print preprocessing details, training progress, model evaluation metrics, and display various plots (confusion matrices, ROC curves).
