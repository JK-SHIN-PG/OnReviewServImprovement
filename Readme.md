# Online Review Mining Meets Interpretable Machine Learning for Customer-Oriented Service Quality Management

### Implementation

- **Our Environment**
  - os : Ubuntu 18.04 (we didn't check Window os environment)
  - Python >= 3.7

```bash
pip install -r requirement.txt
```

```bash
root
├─Params.yaml
├─estimateImportance.py
├─identifyServiceFeatures.py
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
```bash
python estimateImportance.py --yaml="Params.yaml"
```
```bash
python identifyServiceFeatures.py --yaml="Params.yaml"
```