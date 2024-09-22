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




