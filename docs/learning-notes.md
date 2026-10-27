# Learning Notes — Final Year Project

# Step 1: Project Setup and Dataset Inspection

## 1. Project folder setup

**What I did:**

I opened my FYP folder in VS Code and created an `Experiments` folder. Inside it, I created a Python file named `inspect_dataset.py`.

The project structure at this stage is:

```text
FYP/
├── Dataset (PaySim)/
│   └── PS_20174392719_1491204439457_log.csv
└── Experiments/
    └── inspect_dataset.py
```

The dataset folder contains the original CSV file, while the `Experiments` folder contains the Python scripts used for our work.

## 2. Running the Python script

**What is a Python script?**

A Python script is a file containing Python instructions that perform a task when executed.

Our `inspect_dataset.py` script inspects the PaySim CSV and generates a summary of its contents.

**How to run it in VS Code:**

1. Open the FYP folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Make sure the terminal is currently in the FYP root folder.
4. Execute this command:

```bash
python Experiments/inspect_dataset.py
```

The command has three parts:

- `python` tells the system to execute a Python program.
- `Experiments/inspect_dataset.py` specifies the location of the Python file to execute.
- The script then runs and prints its report in the terminal.

The command should be executed from the FYP root folder because the script uses a relative path based on its own location to find the dataset.

## 3. Purpose of the inspection script

Before training machine-learning models, we need to understand the dataset's structure and quality.

The script examines:

1. Total number of rows and columns.
2. Names of all columns.
3. Data types inferred from the first chunk.
4. Number of fraudulent and non-fraudulent transactions.
5. Missing values in each column.
6. Total number of missing values.

## 4. Why chunk-based reading is used

The PaySim CSV is approximately 4.8 GB, so loading the entire file into memory at once could consume a large amount of RAM.

Instead, we use chunk-based reading with a chunk size of 100,000 rows.

For example:

- Chunk 1 processes the first 100,000 rows.
- Chunk 2 processes the next 100,000 rows.
- Subsequent chunks continue until the file has been processed.

The script accumulates the row counts, missing-value counts and fraud-label counts across the chunks, then displays the combined report.

This reduces memory usage because the script processes a smaller portion of the dataset at a time.

## 5. Python libraries used

- **Pathlib:** Constructs the dataset's file path.
- **Collections.Counter:** Counts fraud and non-fraud labels across chunks.
- **Pandas:** Reads and analyses the CSV data.

These libraries provide existing functionality for file paths, counting and data analysis.

## 6. Understanding the script's execution flow

The script performs the following operations in order:

1. Imports the required libraries.
2. Locates the CSV file relative to the script's location.
3. Sets the chunk size to 100,000 rows.
4. Opens the CSV using Pandas chunk-based reading.
5. Processes each chunk in a loop.
6. Counts the rows and missing values.
7. Counts the values in the `isFraud` column.
8. Prints the final dataset report.

## 7. Actual execution results

**Total rows:** 6,362,620

**Total columns:** 11

### Column names and data types

| Column | Data type |
|---|---|
| `step` | int64 |
| `type` | str |
| `amount` | float64 |
| `nameOrig` | str |
| `oldbalanceOrg` | float64 |
| `newbalanceOrig` | float64 |
| `nameDest` | str |
| `oldbalanceDest` | float64 |
| `newbalanceDest` | float64 |
| `isFraud` | int64 |
| `isFlaggedFraud` | int64 |

The data types above were inferred from the first chunk processed by the script.

### Fraud-label distribution

| Label | Meaning | Number of transactions |
|---|---|---:|
| 0 | Non-fraudulent | 6,354,407 |
| 1 | Fraudulent | 8,213 |
| **Total** | | **6,362,620** |

### Missing values

The script reported zero missing values across all 11 columns.

### Libraries and processing behaviour

The dataset was processed in chunks of 100,000 rows. The terminal displayed progress after each 1,000,000 rows, and the inspection completed successfully.

The report confirms that the dataset is highly imbalanced: fraudulent transactions account for approximately 0.129% of all records, while non-fraudulent transactions account for approximately 99.871%.

# Step 2: Exploring Transaction Patterns and Understanding Class Imbalance

## 1. Creating and running the profiling script

**What I did:**

I created a Python file named `profile_dataset.py` inside the `Experiments` folder.

The purpose of this script was to understand the transaction patterns in the PaySim dataset before deciding how to prepare it for machine-learning experiments.

**Command used in the VS Code terminal:**

```bash
python Experiments/profile_dataset.py
```

The script reads the dataset in chunks of 100,000 rows and calculates statistics across the complete dataset.

It displays a sample of 10 transactions, transaction counts by type, fraud counts by transaction type, transaction amount statistics, and the relationship between the fraud label and the existing fraud flag.

## 2. Understanding the transaction sample

The script displayed the first 10 transactions with their column values.

Each row represents a transaction. For example:

- `step`: The simulated time step at which the transaction occurred.
- `type`: The transaction category, such as PAYMENT, TRANSFER or CASH_OUT.
- `amount`: The transaction amount.
- `nameOrig`: The identifier of the originating account.
- `nameDest`: The identifier of the destination account.
- `oldbalanceOrg` and `newbalanceOrig`: The origin account's balance before and after the transaction.
- `oldbalanceDest` and `newbalanceDest`: The destination account's balance before and after the transaction.
- `isFraud`: The fraud label we want to predict.
- `isFlaggedFraud`: An existing flag indicating whether the transaction was flagged by the dataset's original detection process.

The transaction sample helps us understand what individual records look like, while the complete dataset profile helps us identify broader patterns.

## 3. Transaction counts by type

The profiling script reported the following counts:

| Transaction type | Number of transactions |
|---|---:|
| CASH_OUT | 2,237,500 |
| PAYMENT | 2,151,495 |
| CASH_IN | 1,399,284 |
| TRANSFER | 532,909 |
| DEBIT | 41,432 |
| **Total** | **6,362,620** |

CASH_OUT and PAYMENT account for the largest numbers of transactions in the dataset.

These counts describe the distribution of transaction types. They do not, by themselves, indicate which types are more likely to be fraudulent.

## 4. Fraud distribution by transaction type

The profiling results showed:

| Transaction type | Non-fraud | Fraud |
|---|---:|---:|
| CASH_IN | 1,399,284 | 0 |
| CASH_OUT | 2,233,384 | 4,116 |
| DEBIT | 41,432 | 0 |
| PAYMENT | 2,151,495 | 0 |
| TRANSFER | 528,812 | 4,097 |
| **Total** | **6,354,407** | **8,213** |

All fraud-labelled transactions in this dataset belong to the CASH_OUT or TRANSFER categories.

This is an observed pattern in PaySim's simulated data. It should not be interpreted as a universal rule about real-world financial fraud.

## 5. Comparing transaction amounts

The script calculated the following amount statistics:

| Statistic | Non-fraud | Fraud |
|---|---:|---:|
| Number of transactions | 6,354,407 | 8,213 |
| Average amount | 178,197.04 | 1,467,967.30 |
| Minimum amount | 0.01 | 0.00 |
| Maximum amount | 92,445,516.64 | 10,000,000.00 |

The average transaction amount is much higher for the fraud-labelled transactions in this dataset.

However, transaction amount alone cannot reliably determine whether a transaction is fraudulent. A high-value transaction may be legitimate, and a low-value transaction may be fraudulent. We need to evaluate how different features work together.

## 6. Understanding the existing fraud flag

The script compared `isFraud` with `isFlaggedFraud`.

| isFraud | isFlaggedFraud | Number of transactions |
|---:|---:|---:|
| 0 | 0 | 6,354,407 |
| 1 | 0 | 8,197 |
| 1 | 1 | 16 |

Only 16 of the 8,213 fraud-labelled transactions have `isFlaggedFraud = 1`.

This means the existing flag identifies approximately 0.195% of the fraud-labelled transactions.

The two columns serve different purposes:

- `isFraud` is the target label used to train and evaluate our models.
- `isFlaggedFraud` is an existing flag produced by the dataset's original detection process.

We will exclude `isFlaggedFraud` from the initial model inputs so that our experiment does not rely on an existing detection flag.

## 7. Understanding class imbalance

**Question:** What is the condition called when one class has far more records than another, and how do we handle it?

**Answer: Class imbalance.**

In a classification problem, a class represents one of the categories the model must predict.

In our dataset:

- Class 0 represents non-fraudulent transactions.
- Class 1 represents fraudulent transactions.

The dataset contains:

- Non-fraudulent transactions: 6,354,407 (approximately 99.871%).
- Fraudulent transactions: 8,213 (approximately 0.129%).

There are approximately 773 non-fraudulent transactions for every fraudulent transaction. This is a severe class imbalance.

### Why is class imbalance a problem?

A model could predict every transaction as non-fraudulent and still achieve approximately 99.871% accuracy on this dataset.

However, it would fail to detect every fraudulent transaction.

Therefore, high accuracy alone does not prove that a fraud detection model performs well.

### Methods for handling class imbalance

**1. SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE creates synthetic examples of the minority class using existing minority-class examples. This can help a model learn patterns associated with the minority class.

However, the synthetic examples must be appropriate for the data. We should not apply SMOTE blindly.

**2. Class weights**

Where supported, class weights give greater importance to minority-class examples during model training without generating additional records.

**3. Decision-threshold adjustment**

For models that produce scores or probabilities, the decision threshold can be adjusted to change the balance between detecting fraud and generating false alarms. The threshold must be selected using validation data, not the final test set.

**4. Appropriate evaluation metrics**

We will use accuracy along with precision, recall, F1-score, a confusion matrix and precision-recall analysis where appropriate.

- **Precision:** Of the transactions predicted as fraudulent, how many are actually labelled fraudulent?
- **Recall:** Of all transactions labelled fraudulent, how many did the model detect?
- **F1-score:** A combined measure of precision and recall.

### How will we apply imbalance handling correctly?

We must first split the dataset into training and test sets.

Any resampling method, including SMOTE, will be applied only to the training data. The test set must retain its original class distribution and remain separate from the model-training process.

This prevents information from the test set leaking into training and producing overly optimistic results.

## 8. Main observations from this experiment

1. The dataset contains 6,362,620 transactions, with 11 columns.
2. All 8,213 fraud-labelled transactions belong to the CASH_OUT or TRANSFER categories.
3. The average transaction amount is higher for fraud-labelled transactions than for non-fraudulent transactions.
4. The existing `isFlaggedFraud` column flags only 16 of the 8,213 fraud-labelled transactions.
5. The dataset has severe class imbalance, with fraudulent transactions accounting for approximately 0.129% of all records.
6. Accuracy alone is insufficient for evaluating fraud detection because a model can achieve high accuracy while failing to detect fraud.

## 9. Inspecting the time structure of the dataset

### What I did

I created two Python scripts:

- `inspect_time_structure.py`
- `inspect_time_ranges.py`

I ran them using the following commands from the FYP root folder:

```bash
python Experiments/inspect_time_structure.py
python Experiments/inspect_time_ranges.py
```

The first script inspected the time-step range and checked whether transaction rows were ordered by time. The second script counted transactions and fraud-labelled records across five time ranges.

### Results

The dataset contains 743 distinct time steps, ranging from step 1 to step 743. The inspection found no time-order issues in the CSV's row sequence.

The fraud distribution across time ranges was:

| Time range | Transactions | Fraud | Fraud rate |
|---|---:|---:|---:|
| Steps 1–150 | 1,511,368 | 1,710 | 0.1131% |
| Steps 151–300 | 2,562,282 | 1,691 | 0.0660% |
| Steps 301–450 | 1,941,067 | 1,656 | 0.0853% |
| Steps 451–600 | 244,330 | 1,556 | 0.6368% |
| Steps 601–743 | 103,573 | 1,600 | 1.5448% |

### What I learned

The fraud rate is not constant across time. It is much higher in the final time ranges than in the earlier ranges.

This matters when selecting a training and testing split. A chronological split could have a different fraud distribution in its training and test sets.

For the initial comparison of the four individual models and six pairwise combinations, we plan to use a stratified random split with 70% training data, 15% validation data and 15% test data.

Stratification preserves approximately the same class proportions in each subset. The test set will remain untouched during training and model selection.

A separate chronological evaluation may be useful for studying how well the final model generalises to later transactions.

No model has been trained yet, and no model-performance results have been measured at this stage.

# Step 3: Analysing the Time Structure of PaySim

## 1. Purpose

Before splitting the dataset, I examined the `step` column to understand how transactions and fraud cases were distributed over time.

This was important because a random split and a chronological split evaluate different things. A random split measures performance on held-out transactions drawn from a similar overall distribution, whereas a chronological split can help assess performance on later transactions.

## 2. Scripts created

I created two Python files inside the `Experiments` folder:

- `inspect_time_structure.py`
- `inspect_time_ranges.py`

Commands used in the VS Code terminal:

```bash
python Experiments/inspect_time_structure.py
python Experiments/inspect_time_ranges.py
```

## 3. Time-structure results

The first script reported:

- Total transactions: 6,362,620
- First step: 1
- Last step: 743
- Distinct step values: 743
- Detected time-order issues: 0

The CSV rows were in non-decreasing order by the `step` column according to the script's checks.

## 4. Fraud distribution across time ranges

The second script divided the time steps into five ranges.

| Time range | Transactions | Fraud | Fraud rate |
|---|---:|---:|---:|
| Steps 1–150 | 1,511,368 | 1,710 | 0.1131% |
| Steps 151–300 | 2,562,282 | 1,691 | 0.0660% |
| Steps 301–450 | 1,941,067 | 1,656 | 0.0853% |
| Steps 451–600 | 244,330 | 1,556 | 0.6368% |
| Steps 601–743 | 103,573 | 1,600 | 1.5448% |

## 5. What I learned

The fraud rate varies across time ranges. It is considerably higher in the final two ranges than in the earlier ranges.

This means a chronological split could produce training and test sets with substantially different fraud rates.

For the initial model comparison, I chose a stratified random split to preserve approximately the same fraud/non-fraud ratio in the training, validation and test sets.

A separate chronological evaluation may be useful later to investigate how well the model generalises to transactions occurring later in time.

# Step 4: Splitting the Dataset into Training, Validation and Test Sets

## 1. Creating and running the split script

**What I did:**

I created `split_dataset.py` inside the `Experiments` folder.

I ran it using the following command from the FYP root folder:

```bash
python Experiments/split_dataset.py
```

The script read the target column, created stratified splits, wrote three separate CSV files, and checked the row counts and fraud-label distributions.

## 2. Why did we split the dataset?

We need separate datasets for training, validation and final testing.

- **Training set:** Used to train the machine-learning models.
- **Validation set:** Used to compare model settings and tune decision thresholds.
- **Test set:** Reserved for the final evaluation after model selection and tuning are complete.

Keeping the test set separate helps us measure how well a model performs on data that was not used to train or select it.

## 3. Split proportions

We used a 70:15:15 split:

| Subset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

We used stratified sampling based on `isFraud`. This preserves approximately the same proportion of fraud and non-fraud labels in each subset.

The random seed was set to `42` so that the split can be reproduced when the same data and code are used.

## 4. Actual results

| Subset | Total transactions | Fraud | Non-fraud | Fraud percentage |
|---|---:|---:|---:|---:|
| Training | 4,453,833 | 5,749 | 4,448,084 | 0.1291% |
| Validation | 954,394 | 1,232 | 953,162 | 0.1291% |
| Test | 954,393 | 1,232 | 953,161 | 0.1291% |
| **Total** | **6,362,620** | **8,213** | **6,354,407** | **0.1291%** |

The row counts add up to the original dataset size, and the fraud-label counts add up to the original 8,213 fraud cases.

## 5. Output files created

The script created the following files inside `Dataset (PaySim)/splits/`:

```text
Dataset (PaySim)/
└── splits/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

The original CSV was retained unchanged.

## 6. What I learned

Stratified splitting is useful when the target classes are highly imbalanced because it preserves approximately the same class proportions in each subset.

Our split contains the original 0.1291% fraud rate in all three subsets. This allows us to compare models on a consistent class distribution.

The test set must remain separate from model training, preprocessing decisions, model selection and threshold tuning. Its labels should be used only for final evaluation.

# Step 5: Investigating Feature Patterns

## 1. Creating and running the script

I created a Python file named `inspect_feature_patterns.py` inside the `Experiments` folder.

I ran it using this command from the FYP root folder:

```bash
python Experiments/inspect_feature_patterns.py
```

The script reads the training dataset in chunks of 100,000 rows and calculates the average transaction amount and account balances for fraudulent and non-fraudulent transactions, grouped by transaction type.

Only the training set was used for this analysis.

## 2. Why did we inspect these features?

Before selecting the inputs for our machine-learning models, we need to understand how the available features behave.

The numerical features investigated were:

- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

We grouped the transactions by `type` and `isFraud` to compare patterns between fraud and non-fraud cases within each transaction category.

## 3. Actual observations

### CASH_OUT transactions

| Feature | Non-fraud average | Fraud average |
|---|---:|---:|
| Amount | 173,892.28 | 1,435,476.15 |
| Original balance before | 43,410.41 | 1,433,884.24 |
| Original balance after | 17,511.70 | 103.52 |
| Destination balance before | 1,498,276.04 | 1,118,290.24 |
| Destination balance after | 1,689,546.46 | 2,565,995.36 |

The average transaction amount is much higher for fraud-labelled CASH_OUT transactions. The average originating balance after the transaction is also much lower.

### TRANSFER transactions

| Feature | Non-fraud average | Fraud average |
|---|---:|---:|
| Amount | 903,999.49 | 1,499,956.71 |
| Original balance before | 40,832.56 | 1,877,900.84 |
| Original balance after | 7,505.64 | 400,779.89 |
| Destination balance before | 2,587,965.01 | 1,348.84 |
| Destination balance after | 3,580,026.02 | 5,188.76 |

The average transaction amount is higher for fraud-labelled TRANSFER transactions. Their average destination balances are also much lower than those of non-fraudulent transfers.

## 4. What I learned

The balance features and transaction amount show different patterns across fraud and non-fraud groups.

These patterns may help the models distinguish between the two classes. However, the dataset is simulated, so some patterns may reflect how PaySim generates fraudulent transactions rather than general behaviour in real financial systems.

Group averages alone do not establish which features are most predictive. We must evaluate the features through properly controlled experiments.

## 5. Feature-selection decision

For the initial tabular baseline, we will consider the following inputs:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

We will exclude:

- `isFraud`, because it is the target label.
- `isFlaggedFraud`, because it is an existing detection flag.
- `nameOrig` and `nameDest` from the initial tabular model, while preserving the original identifiers for possible graph-based experiments.

The balance columns will remain in the initial baseline, but we will interpret their contribution cautiously because of the simulated nature of the dataset.

This is an initial experimental feature set, not a claim that these features are optimal.

# Step 6: Building the Preprocessing Components

## 1. Creating and running the script

I created a Python file named `prepare_features.py` inside the `Experiments` folder.

I ran it using this command from the FYP root folder:

```bash
python Experiments/prepare_features.py
```

The script fitted a numerical scaler, configured a categorical encoder, saved the preprocessing components and tested their transformations on a sample of 1,000 training records.

## 2. Selected input features

The initial tabular model uses seven original input columns:

**Numerical features:**
- `step`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

**Categorical feature:**
- `type`

The target column is `isFraud`. It is kept separate from the input features because it is the value the models must predict.

The columns `isFlaggedFraud`, `nameOrig` and `nameDest` are excluded from the initial tabular model inputs.

## 3. StandardScaler for numerical features

`StandardScaler` transforms numerical features by subtracting their training-set mean and dividing by their training-set standard deviation.

This helps put numerical features on comparable scales, which is particularly useful for models such as Logistic Regression.

We used `partial_fit()` to learn the scaling statistics in chunks of 100,000 rows rather than loading the entire training set into memory at once.

The scaler was fitted using all 4,453,833 training records.

### Means learned by the scaler

| Feature | Training mean |
|---|---:|
| `step` | 243.3382 |
| `amount` | 179,669.1284 |
| `oldbalanceOrg` | 833,635.1920 |
| `newbalanceOrig` | 854,812.8735 |
| `oldbalanceDest` | 1,100,418.1525 |
| `newbalanceDest` | 1,224,631.7940 |

These are statistics learned from the training set, not model-performance results.

## 4. OneHotEncoder for transaction type

The `type` column contains categorical values rather than continuous numbers.

`OneHotEncoder` represents the five transaction categories using five indicator columns:

- `CASH_IN`
- `CASH_OUT`
- `DEBIT`
- `PAYMENT`
- `TRANSFER`

For example, a TRANSFER transaction is represented by a one in the TRANSFER indicator column and zeros in the other category columns.

The encoder was configured with the five known transaction types and `handle_unknown="ignore"` so that an unexpected category does not cause a prediction-time error.

## 5. Testing the transformations

The script transformed a sample of 1,000 training records.

The output confirmed:

- Scaled numerical data shape: `(1000, 6)`
- Encoded transaction-type data shape: `(1000, 5)`
- Encoded output: sparse representation

Therefore, the initial tabular representation has six numerical features and five encoded categorical features, giving **11 transformed input features**.

This check confirms that the transformations worked on the sample. It does not mean that any fraud detection model has been trained or evaluated.

## 6. Saving the preprocessing components

The script saved the following files:

```text
Experiments/
└── artifacts/
    ├── numeric_scaler.joblib
    ├── type_encoder.joblib
    └── feature_config.json
```

- `numeric_scaler.joblib` stores the fitted numerical scaler.
- `type_encoder.joblib` stores the fitted categorical encoder.
- `feature_config.json` records the feature names, target column, transaction categories and chunk size.

Saving these components allows us to reuse the same transformations instead of fitting new ones independently for validation, testing or future predictions.

The scaler was fitted only on training data. The validation and test data must be transformed using the saved training-fitted components without refitting them.

# Step 7: Verifying the Preprocessing Pipeline

## 1. Creating and running the verification script

I created `verify_preprocessing.py` inside the `Experiments` folder.

I ran it using this command from the FYP root folder:

```bash
python Experiments/verify_preprocessing.py
```

The script loaded the saved scaler and encoder, processed the training, validation and test CSV files in chunks, and checked the transformed feature dimensions and numerical values.

## 2. Verification results

| Dataset | Rows checked | Fraud labels | Input features | Invalid numerical rows |
|---|---:|---:|---:|---:|
| Training | 4,453,833 | 5,749 | 11 | 0 |
| Validation | 954,394 | 1,232 | 11 | 0 |
| Test | 954,393 | 1,232 | 11 | 0 |
| **Total** | **6,362,620** | **8,213** | | **0** |

All rows were transformed successfully. The number of transformed rows matched the number of input rows in every dataset.

## 3. What I learned

The saved preprocessing components can transform all three datasets into the same 11-feature representation.

No NaN or infinite values were found in the scaled numerical features during this check.

The scaler and encoder were fitted using training data and reused on validation and test data without refitting. This avoids fitting preprocessing statistics on the validation or test sets.

This step verifies the preprocessing process. It does not measure fraud detection performance because no model has been trained or evaluated yet.

## Step 8: Train and Evaluate a Logistic Regression-Style Baseline

### Objective
Train a first machine-learning baseline on the PaySim training dataset and evaluate its ability to distinguish fraudulent transactions from genuine transactions.

### Script
`Experiments/train_logistic_regression.py`

### Terminal command
```powershell
python Experiments/train_logistic_regression.py
```

### Implementation details

1. Loaded the saved numerical scaler and transaction-type encoder created during preprocessing.
2. Counted the training labels:
   - Genuine transactions: 4,448,084
   - Fraudulent transactions: 5,749
3. Calculated class weights to reduce the effect of class imbalance:
   - Genuine class: 0.5006
   - Fraud class: 387.3572
4. Used `SGDClassifier` with `loss="log_loss"` and L2 regularization. This is a scalable, incremental logistic-regression-style classifier.
5. Processed the training dataset in chunks of 100,000 rows instead of loading all feature data into memory at once.
6. Saved the trained model as `Experiments/artifacts/logistic_regression.joblib`.
7. Evaluated the model on the validation dataset using a classification threshold of 0.5.

### Validation results

| Metric | Result |
|---|---:|
| Validation transactions | 954,394 |
| Accuracy | 55.87% |
| Genuine transaction precision | 100.00% |
| Genuine transaction recall | 55.81% |
| Fraud precision | 0.29% |
| Fraud recall | 100.00% |
| Fraud F1-score | 0.58% |
| Average precision (AP) | 0.198417 |

### Confusion matrix

| | Predicted genuine | Predicted fraud |
|---|---:|---:|
| Actually genuine | 531,987 | 421,175 |
| Actually fraud | 0 | 1,232 |

### Interpretation

The model detected all 1,232 fraudulent transactions in the validation set, resulting in 100% fraud recall. However, it incorrectly classified 421,175 genuine transactions as fraudulent.

Fraud precision was only 0.29%. This means that, among all transactions flagged as fraud, only a very small proportion were actually fraudulent.

The low accuracy of 55.87% and the large number of false positives show that detecting every fraud case is not sufficient for a useful fraud-detection system.

The training data is highly imbalanced: genuine transactions greatly outnumber fraudulent transactions. Class weighting was introduced to give fraud cases more influence during training, but it did not produce a practical balance between fraud detection and false alerts at the chosen threshold.

Average precision (AP) was 0.198417. AP summarizes precision-recall performance across different score thresholds; it is not the same as accuracy or ROC-AUC.

### Important implementation note

The script uses `SGDClassifier(loss="log_loss")`, rather than scikit-learn's `LogisticRegression` class. It was trained incrementally in one pass through the training file. Because the dataset is ordered by time, this training procedure may affect the result and should be considered when interpreting this baseline.

### Conclusion

The first baseline demonstrates the trade-off between detecting fraud and avoiding false alarms. Its current predictions are not suitable for practical deployment. The results are preliminary validation results, not final test-set results.

## Step 9: Evaluate Different Classification Thresholds

### Objective
Investigate how changing the classification threshold affects fraud precision, fraud recall, F1-score, and false-positive count.

### Script
`Experiments/evaluate_thresholds.py`

### Terminal command
```powershell
python Experiments/evaluate_thresholds.py
```

### Procedure

1. Loaded the saved model, numerical scaler, and transaction-type encoder.
2. Generated prediction scores for all 954,394 validation transactions in chunks of 100,000.
3. Evaluated thresholds of 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995, and 0.999.
4. Compared precision, recall, F1-score, false positives, true positives, and total flagged transactions.

### Observed score percentiles

| Percentile | Prediction score |
|---|---:|
| 50th | 0.0120741 |
| 90th | 0.99097107 |
| 95th | 0.99482504 |
| 99th | 0.99914328 |
| 99.9th | 0.99999999 |

### Results

| Threshold | Precision | Recall | F1-score | False positives | True positives | Flagged |
|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | 0.0029 | 1.0000 | 0.0058 | 421,175 | 1,232 | 422,407 |
| 0.600 | 0.0029 | 1.0000 | 0.0058 | 419,628 | 1,232 | 420,860 |
| 0.700 | 0.0029 | 1.0000 | 0.0059 | 417,698 | 1,232 | 418,930 |
| 0.800 | 0.0030 | 1.0000 | 0.0059 | 414,716 | 1,232 | 415,948 |
| 0.900 | 0.0030 | 1.0000 | 0.0060 | 406,787 | 1,232 | 408,019 |
| 0.950 | 0.0032 | 0.9984 | 0.0064 | 381,945 | 1,230 | 383,175 |
| 0.990 | 0.0100 | 0.8847 | 0.0198 | 107,717 | 1,090 | 108,807 |
| 0.995 | 0.0207 | 0.7735 | 0.0404 | 45,000 | 953 | 45,953 |
| 0.999 | 0.0675 | 0.5593 | 0.1205 | 9,514 | 689 | 10,203 |

### Interpretation

Increasing the threshold reduces the number of transactions classified as fraud. At a threshold of 0.999, false positives fall from 421,175 to 9,514, but fraud recall also falls from 100% to 55.93%.

At this threshold, 689 of the 1,232 actual fraud cases are detected, while 543 are missed. Fraud precision improves to 6.75%, but most flagged transactions are still genuine.

The median score is approximately 0.012, whereas the 90th-percentile score is approximately 0.991. This indicates that a substantial portion of transactions receives high scores.

### Conclusion

Changing the threshold improves the trade-off between fraud detection and false alarms, but no tested threshold makes this baseline clearly practical. Threshold selection alone does not resolve the underlying model-performance problem.

These are validation-set findings. The test set has not been used for this experiment.

## Step 10: Improve the Logistic Regression-Style Baseline with Multiple Epochs

### Objective
Test whether training for multiple epochs and shuffling rows within each chunk improves the model's validation performance.

### Script
`Experiments/train_logistic_regression.py`

### Terminal command
```powershell
python Experiments/train_logistic_regression.py
```

### Changes made
- Trained for three epochs instead of one.
- Shuffled rows within each 100,000-row chunk.
- Kept the same model configuration and preprocessing components.
- Saved the updated model as `Experiments/artifacts/logistic_regression_v2.joblib`.

### Validation results

| Metric | Original model | Three-epoch model |
|---|---:|---:|
| Accuracy | 55.87% | 58.41% |
| Fraud precision | 0.29% | 0.31% |
| Fraud recall | 100.00% | 99.92% |
| Fraud F1-score | 0.58% | 0.62% |
| Average precision (AP) | 0.198417 | 0.413487 |
| False positives | 421,175 | 396,965 |
| True positives | 1,232 | 1,231 |
| False negatives | 0 | 1 |

### Interpretation
The three-epoch model improved accuracy, fraud precision, fraud F1-score, and average precision while reducing false positives by 24,210.

Average precision increased from 0.198417 to 0.413487, suggesting better overall ranking of fraudulent transactions across score thresholds. However, fraud precision at the default threshold remains very low, at approximately 0.31%.

The updated model still produces too many false alarms for practical fraud detection. Shuffling within chunks also does not fully randomize the order of the entire dataset.

### Conclusion
Multiple epochs improved the validation results, but the model remains an imperfect baseline. The comparison uses the same validation split and preprocessing components. The test set has not been used.

## Step 11: Evaluate Thresholds for the Improved Model

### Objective
Evaluate how different classification thresholds affect the improved three-epoch model's fraud-detection performance.

### Script
`Experiments/evaluate_thresholds.py`

### Terminal command
```powershell
python Experiments/evaluate_thresholds.py
```

The script was updated to load `Experiments/artifacts/logistic_regression_v2.joblib`.

### Results

| Threshold | Precision | Recall | F1-score | False positives | True positives | Flagged |
|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | 0.0031 | 0.9992 | 0.0062 | 396,965 | 1,231 | 398,196 |
| 0.600 | 0.0033 | 0.9984 | 0.0065 | 375,994 | 1,230 | 377,224 |
| 0.700 | 0.0045 | 0.9959 | 0.0089 | 274,045 | 1,227 | 275,272 |
| 0.800 | 0.0115 | 0.9448 | 0.0226 | 100,438 | 1,164 | 101,602 |
| 0.900 | 0.0408 | 0.7922 | 0.0775 | 22,973 | 976 | 23,949 |
| 0.950 | 0.1290 | 0.6494 | 0.2152 | 5,403 | 800 | 6,203 |
| 0.990 | 0.4307 | 0.4562 | 0.4430 | 743 | 562 | 1,305 |
| 0.995 | 0.5340 | 0.3953 | 0.4543 | 425 | 487 | 912 |
| 0.999 | 0.6590 | 0.3247 | 0.4350 | 207 | 400 | 607 |

### Interpretation

Increasing the threshold generally reduces false positives but also reduces fraud recall. At threshold 0.995, precision is 53.40%, recall is 39.53%, and the F1-score is 0.4543.

Among the tested thresholds, 0.995 gives the highest F1-score. At this threshold, 487 of 1,232 fraud cases are detected, while 425 genuine transactions are incorrectly flagged.

The threshold is selected only for analysis on the validation set. It has not been tested on the held-out test set and should not be treated as a deployment-ready setting.

### Conclusion

Threshold adjustment substantially improves the precision-recall balance compared with the default threshold of 0.5. Nevertheless, the model still misses many fraudulent transactions, so further model comparisons are necessary.

## Step 12: Train and Evaluate a Random Forest Baseline

### Objective
Train a Random Forest classifier on a sample of the PaySim training data and evaluate its ability to detect fraudulent transactions.

### Script
`Experiments/train_random_forest.py`

### Terminal command
```powershell
python Experiments/train_random_forest.py
```

### Training procedure
- Sampled 10% of each training-data chunk.
- Total sampled rows: 445,383.
- Genuine transactions in sample: 444,803.
- Fraudulent transactions in sample: 580.
- Applied the existing numerical scaler and transaction-type encoder.
- Trained a Random Forest with 100 trees, maximum depth 20, minimum leaf size 2, and `balanced_subsample` class weighting.
- Saved the model as `Experiments/artifacts/random_forest.joblib`.
- Evaluated on the complete validation set of 954,394 transactions.

### Validation results

| Metric | Result |
|---|---:|
| Accuracy | 99.9655% |
| Fraud precision | 97.88% |
| Fraud recall | 74.92% |
| Fraud F1-score | 0.8487 |
| Average precision (AP) | 0.887517 |
| True positives | 923 |
| False positives | 20 |
| True negatives | 953,142 |
| False negatives | 309 |

### Confusion matrix

| | Predicted genuine | Predicted fraud |
|---|---:|---:|
| Actually genuine | 953,142 | 20 |
| Actually fraud | 309 | 923 |

### Interpretation
The Random Forest correctly identified 923 of the 1,232 fraudulent validation transactions and incorrectly flagged only 20 genuine transactions. It missed 309 fraud cases.

Fraud precision of 97.88% means that most transactions flagged by this model were genuinely fraudulent within this validation dataset. Fraud recall of 74.92% means that it detected approximately three-quarters of the actual fraud cases.

Average precision was 0.887517, indicating strong precision-recall performance across score thresholds.

These results are substantially better than the initial Logistic Regression-style baseline. However, the Random Forest was trained on a 10% sample, and PaySim is a synthetic dataset. These results should not be interpreted as evidence of equivalent performance on real banking transactions.

### Conclusion
The Random Forest is a promising baseline for this project. Its validation results show a better balance between detecting fraud and limiting false alarms than the initial Logistic Regression-style model. The held-out test set has not been used.

## Step 13: Train and Evaluate XGBoost

### Objective
Train an XGBoost classifier using the same sampled training data and validation split as Random Forest, then compare their fraud-detection performance.

### Script
`Experiments/train_xgboost.py`

### Terminal command
```powershell
python Experiments/train_xgboost.py
```

### Training procedure
- Sampled 10% of each training-data chunk.
- Training sample: 445,383 transactions, including 580 fraud cases and 444,803 genuine cases.
- Used the existing numerical scaler and transaction-type encoder.
- Configured XGBoost with 300 estimators, maximum depth 8, learning rate 0.1, histogram-based tree construction, and `scale_pos_weight` to address class imbalance.
- Saved the model as `Experiments/artifacts/xgboost.joblib`.
- Evaluated on all 954,394 validation transactions at a classification threshold of 0.5.

### Validation results

| Metric | Result |
|---|---:|
| Accuracy | 99.9623% |
| Fraud precision | 84.44% |
| Fraud recall | 86.77% |
| Fraud F1-score | 0.8559 |
| Average precision (AP) | 0.925884 |
| True positives | 1,069 |
| False positives | 197 |
| True negatives | 952,965 |
| False negatives | 163 |

### Comparison with Random Forest

Both models used the same 10% training sample and validation split.

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Fraud precision | 97.88% | 84.44% |
| Fraud recall | 74.92% | 86.77% |
| Fraud F1-score | 0.8487 | 0.8559 |
| Average precision | 0.887517 | 0.925884 |
| False positives | 20 | 197 |
| False negatives | 309 | 163 |

### Interpretation
XGBoost detected 1,069 fraud cases, compared with 923 for Random Forest. It missed 163 fraud cases, compared with 309 for Random Forest, but produced 197 false positives instead of 20.

XGBoost achieved a higher fraud F1-score and average precision, while Random Forest achieved higher precision at the default threshold.

### Conclusion
XGBoost is a strong baseline in this experiment, particularly in fraud recall and average precision. The comparison is preliminary: both models were trained on a 10% sample, and PaySim is synthetic. The held-out test set has not been used.

## Step 14: Train and Evaluate Isolation Forest

### Objective
Evaluate an anomaly-detection model and compare its ability to identify fraudulent transactions with the supervised models.

### Script
`Experiments/train_isolation_forest.py`

### Terminal command
```powershell
python Experiments/train_isolation_forest.py
```

### Training procedure
- Selected genuine transactions from the training split only.
- Sampled 5% of genuine transactions from each chunk, resulting in 222,402 training rows.
- Applied the existing numerical scaler and transaction-type encoder.
- Trained Isolation Forest with 100 trees, `max_samples=256`, and `contamination="auto"`.
- Saved the model as `Experiments/artifacts/isolation_forest.joblib`.
- Scored all 954,394 validation transactions.
- Classified approximately the dataset's overall fraud proportion as anomalous, using an anomaly-score threshold of 0.706602.

### Validation results

| Metric | Result |
|---|---:|
| Accuracy | 99.7450% |
| Fraud precision | 1.30% |
| Fraud recall | 1.30% |
| Fraud F1-score | 0.0130 |
| Average precision (AP) | 0.007702 |
| True positives | 16 |
| False positives | 1,218 |
| True negatives | 951,944 |
| False negatives | 1,216 |

### Confusion matrix

| | Predicted genuine | Predicted fraud |
|---|---:|---:|
| Actually genuine | 951,944 | 1,218 |
| Actually fraud | 1,216 | 16 |

### Interpretation
Isolation Forest detected only 16 of the 1,232 fraudulent validation transactions. Its fraud precision and recall were both approximately 1.30%.

Although accuracy was 99.7450%, this metric is misleading in a highly imbalanced dataset because genuine transactions make up almost all the data.

Anomaly detection identifies unusual observations, but unusual behaviour is not necessarily fraudulent. Conversely, some fraudulent transactions may not appear unusual.

### Conclusion
Isolation Forest performed poorly for fraud detection under this experiment's settings. Its results are not directly comparable to supervised models without considering the different training methods and threshold choices. The held-out test set has not been used.

## Step 15: Compare All Four Baseline Models

### Objective
Compare the four saved models on the same validation dataset using accuracy, fraud precision, fraud recall, F1-score, average precision, false positives, and false negatives.

### Script
`Experiments/compare_models.py`

### Terminal command
```powershell
python Experiments/compare_models.py
```

### Results

| Model | Accuracy | Fraud precision | Fraud recall | Fraud F1 | AP | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression-style | 0.5841 | 0.0031 | 0.9992 | 0.0062 | 0.4135 | 396,965 | 1 |
| Random Forest | 0.9997 | 0.9788 | 0.7492 | 0.8487 | 0.8875 | 20 | 309 |
| XGBoost | 0.9996 | 0.8444 | 0.8677 | 0.8559 | 0.9259 | 197 | 163 |
| Isolation Forest | 0.9974 | 0.0130 | 0.0130 | 0.0130 | 0.0077 | 1,218 | 1,216 |

### Interpretation
XGBoost achieved the highest average precision (0.9259), highest fraud recall among these default-threshold supervised predictions (86.77%), and highest fraud F1-score (0.8559).

Random Forest achieved higher precision and generated fewer false positives at the selected threshold of 0.5. Isolation Forest performed poorly at the chosen prevalence-based anomaly threshold.

The comparison is not perfectly controlled for training-data size: the Logistic Regression-style model used the full training split, while Random Forest and XGBoost used 10% samples. Isolation Forest used a 5% sample of genuine training transactions.

### Conclusion
XGBoost is the current leading candidate based on F1-score and average precision, while Random Forest is a useful alternative when precision and low false-alarm counts are priorities. These are preliminary validation results on synthetic PaySim data; the held-out test set has not been used.

## Step 16: Compare Individual Models and Six Pairwise Ensembles

### Objective
Evaluate all four individual models and all six two-model combinations using the same validation transactions and a consistent fixed-size ranking procedure.

### Script
`Experiments/compare_pairwise_ensembles.py`

### Terminal command
```powershell
python Experiments/compare_pairwise_ensembles.py
```

### Method
- Loaded the four saved models and the existing preprocessing components.
- Generated scores for all 954,394 validation transactions.
- Converted each model's scores to percentile ranks so different score scales could be combined.
- Created all six pairwise combinations by averaging the percentile ranks of two models.
- Selected 1,232 transactions per model or combination as predicted fraud, using the dataset's overall fraud prevalence to set the fixed selection count.
- Calculated accuracy, precision, recall, F1-score, average precision, false positives, and false negatives.
- Saved the results to `Experiments/artifacts/pairwise_ensemble_results.csv`.

### Results

| Model / combination | Accuracy | Precision | Recall | F1-score | AP |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 99.8569% | 44.56% | 44.56% | 0.4456 | 0.413487 |
| Random Forest | 99.9547% | 82.47% | 82.47% | 0.8247 | 0.887517 |
| XGBoost | 99.9631% | 85.71% | 85.71% | 0.8571 | 0.925884 |
| Isolation Forest | 99.7452% | 1.30% | 1.30% | 0.0130 | 0.007702 |
| Logistic Regression + Random Forest | 99.9300% | 72.89% | 72.89% | 0.7289 | 0.785972 |
| Logistic Regression + XGBoost | 99.9235% | 70.37% | 70.37% | 0.7037 | 0.761203 |
| Logistic Regression + Isolation Forest | 99.7802% | 14.85% | 14.85% | 0.1485 | 0.087259 |
| Random Forest + XGBoost | 99.9612% | 84.98% | 84.98% | 0.8498 | 0.916096 |
| Random Forest + Isolation Forest | 99.8525% | 42.86% | 42.86% | 0.4286 | 0.444752 |
| XGBoost + Isolation Forest | 99.8233% | 31.57% | 31.57% | 0.3157 | 0.272470 |

### Interpretation
XGBoost achieved the highest average precision (0.925884) and F1-score (0.8571) in this fixed-size ranking experiment.

The Random Forest + XGBoost combination achieved an F1-score of 0.8498 and AP of 0.916096, both below the corresponding XGBoost-only results. The other pairwise combinations also did not outperform XGBoost on these metrics.

Because exactly 1,232 transactions were selected as fraud predictions and the validation set contains 1,232 actual fraud cases, precision and recall are equal by construction in this experiment. These values should therefore be interpreted as fixed-size ranking results, not ordinary deployment-threshold metrics.

### Conclusion
Under the tested rank-averaging method, no pairwise combination outperformed XGBoost alone on F1-score or average precision. XGBoost is the leading individual baseline based on this experiment.

The results are preliminary, use synthetic PaySim data, and have not been evaluated on the held-out test set.

# Step 8: Chronological Split — Time-Based Dataset Preparation

### Objective
Create a separate chronological train, validation and test split from the original PaySim dataset to support time-aware fraud-detection experiments.

### Command executed
```powershell
python Experiments/split_dataset_chronological.py
```

### What the script did
- Read the original PaySim CSV in chunks of 100,000 rows.
- Assigned transactions to three non-overlapping time ranges.
- Preserved the original columns and transactions within each range.
- Saved the new splits under `Dataset (PaySim)/chronological_splits/`.
- Verified the combined row count and checked for overlapping steps.

### Observed results

| Split | Rows | Fraud cases | Fraud rate | Step range |
|---|---:|---:|---:|---|
| Training | 6,014,717 | 5,057 | 0.0841% | 1–450 |
| Validation | 244,330 | 1,556 | 0.6368% | 451–600 |
| Test | 103,573 | 1,600 | 1.5448% | 601–743 |

### Verification
- Original CSV rows: 6,362,620
- Combined chronological-split rows: 6,362,620
- Combined row count matched the original: True
- Overlapping steps between splits: None
- Rows outside steps 1–743: 0

### Output files
- `Dataset (PaySim)/chronological_splits/train.csv`
- `Dataset (PaySim)/chronological_splits/validation.csv`
- `Dataset (PaySim)/chronological_splits/test.csv`

### Interpretation
The chronological split separates earlier transactions for training, a middle period for validation and a later period for testing. The fraud rate increases across these periods, reaching 1.5448% in the test period compared with 0.0841% in training.

This indicates a substantial difference in fraud prevalence across time. Models evaluated on these splits must be assessed with this distribution difference in mind.

The chronological splits are separate from the original random splits. The existing random splits, preprocessing artifacts and trained baseline models were not modified.

This step prepared and verified the datasets. No model was trained or evaluated using the chronological splits in this experiment.

# Step 9: Inference-Only Web Application

### Implementation
- Added a FastAPI service in `backend/main.py` that loads the existing chronological main-feature XGBoost model and saved transaction-type encoder at startup.
- Inspected the saved artifacts before implementation: the encoder categories are `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`, its output type is `float32`, and the classifier expects 11 features.
- The API validates six numeric fields and one transaction type, then assembles six `float32` numeric columns in training order followed by the five encoded type columns.
- Added `/api/health` and `/api/predict` endpoints. The prediction endpoint applies the saved validation threshold of 0.95 and labels the result as a model prediction, not a guarantee.
- Added `frontend/index.html` with a transaction form, connection status, loading and error states, and score/decision panel. It identifies the system as a research prototype using synthetic PaySim data.
- Added setup and run instructions to the root `README.md`.

### Scope and limitation
The web application loads the existing artifacts and does not retrain models or read train, validation, or test datasets during inference. The no-balance ablation performed substantially worse in the saved experiment results (test F1 0.3005 versus 0.8804 for the main model); this limitation is documented in the README. No application test outcomes are recorded here until verification is run.

# Step 10: Integrated Payment Trust Lab Prototype

### Components added
- Added SQLite-backed local demo accounts and user-scoped history in `backend/persistence.py`. Passwords are pre-hashed with SHA-256 and bcrypt-hashed; no plaintext password is stored.
- Extended the API with registration, login, current-user, private history, local account-history graph summary, and URL indicator routes. Existing transaction model inputs and anonymous scoring behavior are preserved; optional account IDs are metadata only.
- Added `backend/url_analysis.py`, a network-free URL text and hostname indicator checklist. This is not a trained URL classifier and does not establish safety.
- Updated the frontend with Overview, Transaction Assessment, Account / Graph Analysis, QR / URL Check, and History sections. No actual bank or UPI connection is made.
- Added `Experiments/train_temporal_graph_features.py` as a reproducible graph-feature XGBoost candidate. It uses only graph state from earlier `step` values, excludes fraud labels from graph features, samples training rows consistently with the existing chronological XGBoost procedure, selects its threshold on validation, and writes new outputs to a separate artifact directory.
- The graph feature state uses fixed-size Count-Min Sketches rather than storing every account key, to bound memory. Hash collisions can overestimate degree counts and are a stated limitation.
- Added `docs/architecture.md` and updated `README.md` with component boundaries, reproduction commands, contributions, setup, and limitations.

### Environment and research limitations
The inspected environment does not include PyTorch, PyTorch Geometric, DGL, or NetworkX, so this project implements a graph-feature baseline rather than claiming a GNN. No URL dataset or QR decoding library is present; URL analysis is local rules and direct text input only. The full temporal graph experiment must complete before any graph-improvement claim is made. Interactive account graph summaries use only the current user's saved app assessments and are separate from offline PaySim graph research.

### Verification performed
- Python syntax compilation passed for the backend and graph experiment script.
- `python Experiments/train_temporal_graph_features.py --self-test` passed its synthetic same-step leakage check.
- A localhost API integration run passed dashboard routing, registration, login, wrong-password rejection, authenticated transaction and URL saves, user-scoped history isolation, account-history relationship summary, URL indicator check, and denial of unauthenticated history access.
- The full graph-feature experiment completed successfully. Its validation-selected threshold was 0.88. Chronological test metrics: precision 0.9776, recall 0.7925, F1 0.8754, average precision 0.9605. Compared with the saved main XGBoost on that same test split (precision 0.9749, recall 0.8025, F1 0.8804, average precision 0.9639), the candidate did not improve F1 or average precision. No graph-performance improvement is claimed.

# Step 11: Reproducible Demonstration and Research Comparison

### Implementation
- Added `frontend/research_demo_transactions.json`, a separate file with five fictional accounts and six unlabeled PaySim-shaped transaction scenarios. The same rows power a directed SVG relationship view and the sample selector; no existing dataset, user history, split, or trained artifact is used as demo data.
- Added `demo/run_model_scenarios.py`. It creates a uniquely named temporary SQLite database, confirms the backend selected it and the existing chronological model, and sends each row through the actual FastAPI `/api/predict` route without authentication or saved history. It also checks documented URL rules locally without visiting URLs.
- Recorded the actual returned prediction objects in `demo/scenario_outputs.json`. All six illustrative model decisions were “Likely legitimate” under the fixed 0.95 threshold. These are model outputs for fictional inputs, not verified ground-truth labels or model performance metrics.
- Added a side-by-side Research Results comparison using the existing main XGBoost and temporal graph-feature JSON results. The graph-feature candidate did not improve F1 or average precision and is not a GNN.
- Added URL-rule examples, graph-scope distinctions, reproduction commands, observed sample outputs, and QR-decoder feasibility to `docs/demo-guide.md`.

### Verification and limitations
- `python demo/run_model_scenarios.py` passed using a temporary database that was removed after the run. The output file records all six actual API responses.
- The URL cases cover non-HTTPS, user information, literal IP host, punycode, known shortener, unusual port, and sensitive terms. The script makes no network requests for these checks.
- The environment has Pillow but no `cv2`, `pyzbar`, `zxingcpp`, or `zbarimg`; QR image decoding remains unsupported. No package was installed.
- Demo graph rows are fictional and unlabeled. They are separate from authenticated per-user history and the offline past-only PaySim graph features.
