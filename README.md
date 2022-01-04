# Missing_data
Objective: This Project tries applying imputation methods on missing data. We want to show others that missing data imputation method is better than simply listwise.

Processes:
  1. Download a complete financial dataset from WRDS. This data is stored at 'data/complete_dataset.csv' (10 variables)
  2. Produce three types of missing datasets: MCAR, MAR, and MNAR.
  3. Apply Machine Learning (ML) Imputation method to impute missing data.
  4. Apply traditional method of dealing missing data in finance department (lisewise: delete observation if any variable's value in this observation is missing).
  5. See if ML Imputation method is better than lisewise by KS test (distrbution comparision test).

Codes:
  Just excute the produce_NA.py, you will finally get all the KS test results. (hint: follow instructions in produce_NA.py)
