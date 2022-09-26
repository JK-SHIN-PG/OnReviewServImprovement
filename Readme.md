# Online Review Mining Meets Interpretable Machine Learning for Customer-Oriented Service Quality Management

## Explanation
- Under construction

## Implementation

- **Our Environment**
  - os : Ubuntu 18.04
  - Python == 3.7  

we recommend to create new virtual environment. 

```bash
conda create -n 'env_name' python=3.7 
conda activate 'env_name'
# (or) source activate 'env_name'
```

```bash
git clone https://github.com/ServEngKD/OnReviewServImprovement.git
cd OnReviewServImprovement/
```

Run the below code to install the required package for this implementation.
```bash
#(if you need) pip install --upgrade pip
pip install -r requirements.txt
```

### Code structure
```bash
root
├─Params.yaml
├─identifyServiceFeatures.py
├─estimateImportance.py
├─src
│   ├─ __init__.py
│   ├─ main_components.py
│   ├─ preprocessors.py
│   └─ utils.py
├─Datasets
│   └─ Custom
│      └─***.** (filename.type)
└─Results
    ├─ [1]LDA - Result_Folder_for_each_combination
    ├─ [2]ML - Hyper_parameter_tuning_Reports
    ├─ [3]IPA - Plot_for_Importance_Performance_Analysis
    └─ etc.
```
## Note
Before you run the below code, please check the `Params.yaml` file.
In `Params.yaml`, you can set the hyper-parameters for our framework and custom data path.
In this implementation, grid search was performed to find optimal LDA and ML models. So, you have to set the candidates of each hyperparameter of both models.
Also, you can change the name of `.yaml` file. If you change it, please enter the file name at `--ymal="ymal_name"` in the below codes.

If you want to implement this framework for your review datasets (star rating is required), Modify `CustomizedPreprocessor` in `/src/preprocessors.py` and `CUSTDATA_PATH` in `Params.yaml`.

## [1] Review data preprocessing and service feature identification
- Execution list
  * Text preprocessing
  * LDA Topic modeling

Running the code below generates preprocessed datasets in `/Results` folder and a `summary.txt` and report with a words-topic list for each combination of candidates in each folder in `/Results/[1]LDA/`.
```bash
python identifyServiceFeatures.py --yaml="Params"
```

Referring to the `summary.txt` and words-topic list in `report.txt`, choose the best results of LDA topic modeling and name each topic.
Then, Enter the folder name at `LDA_RESULT_IDX` and the name of each topic in `TopicList` in `yaml`file and 

## [2~3] Global importance estimation using optimal prediction model
- Execution list
  * Preparing the datasets for training (creating review-feature matrix and spliting datasets)
  * K cross validation for hyper-parameter tuning and finding optimal prediction model
  * Estimation of global importance of service features
  * Importance Performance Analysis (plot)

Running the code below generates `report.txt` that record the performance of each model in `/[2]ML` folder and saves the plot for Importance Performance Analysis (IPA) in `/[3]IPA` folder.
```bash
python estimateImportance.py --yaml="Params"
```

