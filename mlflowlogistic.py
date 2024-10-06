import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
# from mlflow_utils import get_mlflow_experiment

dataset=pd.read_csv("cleansed_dataset.csv")
x=dataset.drop('Class',axis=1)
y=dataset['Class']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#mlflow.delete_experiment(experiment_id="746035795611261080")
#mlflow.tracking.MlflowClient().delete_experiment(experiment_id="746035795611261080")
# start writing the model using mlflow
# experiment_id = mlflow.create_experiment(
#         name="logistic_regression_v_1",
#         artifact_location="logistic_regression_artifacts",
#     )
# experiment = get_mlflow_experiment(experiment_id="744107063558873652")
for i in range(500, 5001, 500) :
    with mlflow.start_run(run_name=f'logistic_regression_v_{i}', experiment_id = '744107063558873652') as run:
        log_model=LogisticRegression(max_iter=i) # logistic regression parameter max iteration
        log_model.fit(x_train,y_train)
        mlflow.log_param('max_iter', i)
        y_pred=log_model.predict(x_test)
        x_train_pred=log_model.predict(x_train)
        test_scores = accuracy_score(y_pred,y_test)
        train_scores = accuracy_score(x_train_pred,y_train)
        mlflow.log_metric('test_accuracy', test_scores)
        mlflow.log_metric('train_accuracy', train_scores)
        conf_matrix=confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(conf_matrix,annot=True,fmt='d', cmap='Blues',xticklabels=['Non_fraud_predicted','Fraud_predicted'],yticklabels=['Actual_Non_Fraud','Actual_Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(f'confusion_matrix_{i}.png')
        mlflow.log_artifact(f'confusion_matrix_{i}.png')
        mlflow.sklearn.log_model(log_model, f'logistic_regression_model_{i}')
