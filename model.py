# %%
import pandas as pd

# %%
dataset=pd.read_csv("cleansed_dataset.csv")
dataset.head()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# %%
print(list(dataset.columns))


# %%
x=dataset.drop('Class',axis=1)
x

# %%
y=dataset['Class']
y

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# %%
log_model=LogisticRegression(max_iter=1000)

# %%
log_model.fit(x_train,y_train)


# %%
y_pred=log_model.predict(x_test)
y_pred

# %%
x_train_pred=log_model.predict(x_train)
x_train_pred

# %%
accuracy_score(y_pred,y_test) #for test data

# %%
accuracy_score(x_train_pred,y_train)

# %%
conf_matrix=confusion_matrix(y_test,y_pred)
conf_matrix

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix,annot=True,fmt='d', cmap='Blues',xticklabels=['Non_fraud_predicted','Fraud_predicted'],yticklabels=['Actual_Non_Fraud','Actual_Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show

# %%
print(classification_report(y_test,y_pred))

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
RF_model=RandomForestClassifier(n_estimators=100,random_state=42)

# %%
RF_model.fit(x_train,y_train)

# %%
y_pred_rf=RF_model.predict(x_test)
y_pred_rf

# %%
x_train_pred_RF=RF_model.predict(x_train)

# %%
accuracy_score(y_pred_rf,y_test)

# %%
accuracy_score(x_train_pred_RF,y_train)

# %%
conf_matrix_RF=confusion_matrix(y_test,y_pred_rf)
conf_matrix_RF

# %%
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix_RF,annot=True,fmt='d', cmap='Blues',xticklabels=['Non_fraud_predicted','Fraud_predicted'],yticklabels=['Actual_Non_Fraud','Actual_Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show

# %%
from sklearn.model_selection import cross_val_score,KFold

# %%
kf=KFold(n_splits=5,shuffle=True,random_state=42)
Cross_Scores=cross_val_score(RF_model,x,y,cv=kf,scoring='accuracy')
Cross_Scores

# %%
Cross_Scores.mean()

# %%
print(classification_report(y_test,y_pred_rf))


