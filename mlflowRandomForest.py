import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("cleansed_dataset.csv")
x = dataset.drop('Class', axis=1)
y = dataset['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

experiment_id = mlflow.create_experiment(
    name="random_forest_v_1",
    artifact_location="random_forest_artifacts",
)
l = [50, 100, 200, 500, 1000, 2000]

for i in l:
    with mlflow.start_run(run_name=f'random_forest_v_{i}', experiment_id=experiment_id) as run:
        RF_model = RandomForestClassifier(n_estimators=i, random_state=42)
        RF_model.fit(x_train, y_train)
        mlflow.log_param('n_estimators', i)
        y_pred_rf = RF_model.predict(x_test)
        x_train_pred_RF = RF_model.predict(x_train)
        test_scores = accuracy_score(y_pred_rf, y_test)
        train_scores = accuracy_score(x_train_pred_RF, y_train)
        mlflow.log_metric('test_accuracy', test_scores)
        mlflow.log_metric('train_accuracy', train_scores)
        conf_matrix_RF = confusion_matrix(y_test, y_pred_rf)

        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix_RF, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non_fraud_predicted', 'Fraud_predicted'],
                    yticklabels=['Actual_Non_Fraud', 'Actual_Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        plt.savefig(f'confusion_matrix_RF_{i}.png')
        mlflow.log_artifact(f'confusion_matrix_RF_{i}.png')
        mlflow.sklearn.log_model(RF_model, f'random_forest_model_{i}')
