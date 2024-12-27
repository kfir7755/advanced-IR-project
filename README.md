# README

## Installation
To install the required package, run the following command:
```bash
pip install trectools
```

## Data Folder Structure
The data folder should be organized as follows:
```
data/
├── robust/
    ├── robust03_qrels.txt
├── trec3.results.input.tar
```
### Note
The data files are not uploaded to this GitHub repository as publishing them is not permitted. However, the QRELs file can be downloaded directly from the following link:
[robust03_qrels.txt](https://trec.nist.gov/data/qrels_eng/robust03_qrels.txt)

## Steps to Run
1. **Generate Per Query QRELs File:**
   Run the `per query qrels file.py` script to create QRELs text files for individual queries.
   ```bash
   python "per query qrels file.py"
   ```

2. **Main File:**
   Use the main file to generate results.
   ```bash
   python main.py
   ```

3. **Create Folds:**
   Run the `create folds.py` script to create 5 folds (or more accurately cross-test folds) for cross-validation.
   ```bash
   python "create folds.py"
   ```

