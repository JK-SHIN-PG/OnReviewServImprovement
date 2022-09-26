# Online Review Mining Meets Interpretable Machine Learning for Customer-Oriented Service Quality Management

### Implementation

- **Our Environment**
  - os : Ubuntu 18.04 (we didn't check Window os environment)
  - Python >= 3.7

Run the below code to set the environment
```bash
pip install -r requirement.txt
```
- Code structure
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
│      └─***.**
└─Results
    ├─ [1]LDA
    ├─ [2]ML 
    ├─ [3]IPA
    └─ etc.
```

## [1~2] Review data preprocessing and service feature identification

```bash
python identifyServiceFeatures.py --yaml="Params"
```

## [3] Global importance estimation using optimal prediction model
```bash
python estimateImportance.py --yaml="Params"
```

