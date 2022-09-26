# Online Review Mining Meets Interpretable Machine Learning for Customer-Oriented Service Quality Management

## Explanation
- Under construction

## Implementation

- **Our Environment**
  - os : Ubuntu 18.04 (we didn't check Window os environment)  
  - Python >= 3.7  

we recommend to create new virtual environment. 

```bash
conda create -n 'env_name' python=3.7 
```

Run the below code to install the required package for this implementation.
```bash
pip install -r requirement.txt
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
Before you run the below code, please check the `Params.yaml` file
In `Params.yaml`, you can set the hyper-parameters for our framework.
In this implementation, grid search was performed to find optimal LDA and ML models. So, you have to set the candidates of each hyperparameter of both models.
Also, you can change the name of `.yaml` file. If you change it, please enter the file name at `--ymal="ymal_name"` in the below codes.

## [1] Review data preprocessing and service feature identification

```bash
python identifyServiceFeatures.py --yaml="Params"
```
After running the code, we have to interpret the results of LDA and name each topics.
You have to enter the name of each topic in `TopicList` in `yaml`file

## [2~3] Global importance estimation using optimal prediction model
```bash
python estimateImportance.py --yaml="Params"
```

