# Credit Card Fraud Detetction

- `Credit Card Fraud Detection` is the chosen graduation project idea in DEPI (Digital Egypt Pioneers Initiative) by the teammates.
  
# Table Of Contents:

<Details><summary> Click To Expand</summary>

1. [About The Idea](#1--about-the-idea)
2. [The Dataset](#2--the-dataset)
3. [Project WorkFlow](#3--project-workflow)
4. [Data Preprocessing]()
5. [Model Selection And Training]()
6. [Inference And Evaluation]()
7. [Model Deployment]()
8. [Workflow]()
9.  [Acknowledgements]()

</Details>

----

## 1- About The Idea:

Credit card fraud is a major source of financial trouble not only for consumers, but for banks too. 

Credit card fraud detection is a set of tools and protocols which card issuers use to detect suspicious activity that could indicate a fraud attempt. These tools are generally proactive, aiming to stop credit card fraud before it starts. They also help to prevent financial losses caused by credit card fraud

### Why It's important?

Credit Card Fraud Detection helps to prevent financial losses caused by credit card fraud.

## 2- The Dataset:

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. [The dataset link and read more about it here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## 3- Project WorkFlow

Project workflow is as follows:

![Project Workflow](/project%20workflow.png)

Data was collected and downloaded from [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Setting up the project worksapce on local pc:

1.  unzipped the dataset and extracted the csv file from the directory
2. Initialized the local directory into a git directory using the command 

```sh
    git init
```

3. making sure that: I'm in the correct directory using one of the two ways navigating using `cd` command followed by the path or directly from the shortcut menu use the git bash here.

4. rename the `master` branch into `main` branch using the command:
   
```sh
    git branch -m master main
```

5. created repository on GitHub have the same name as the problem and connected the local repo and the cloud repo together using 
```sh
    git remote add origin 'url' 
```
6. created the preprocessing notebook.

## 4- Preprocessing

Before feeding the data into the model directly, the data preprocessing step is done first. The used preprocessing techniques used:

1. Ensuring the data is clean (no null values, outliers, ...etc)
2. Checking the class imabalance (solved using sampling)
3. PCA was already applied on the data, so we ensured that all the values are normalized using the standard scaler for better training.
4. exported the cleansed data

### 1. Cleansing the data:
   
   - After importing the dataset in the notebook using `pandas`, we gained some insights about the dataset using `.info()` function as follows:
    
   ![dataset insights](/images/data%20insights.png) 

   - Only the target column is of type integer and the rest of the columns are of float type, also they are all have the same count (no missing values), you can check for null values using `df.isnull().sum()` to sum the null vaules of each column.
   - The dataset is too large, we have to check to class imabalnce to prevent bias:
   

   ![class imabalance](/images/class%20imbalance.png)


   - After visualizing the calss balance, we found that one of two classes is too little to the other class, so a class imbalance problem is addressed here to be solved.
   
   - There are multiple techniques to resolve class imbalance: GANs, SMOTE, resampling, ...etc. We used `resampling` using `scikit-learn` library.






