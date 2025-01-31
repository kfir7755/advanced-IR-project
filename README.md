# README

## Installation
To install the required package, run the following command:
```bash
pip install trectools tqdm
```
if you wish to get a .tex version of the results also run the following command:
```bash
pip install Jinja2
```
## Data Folder Structure
The data folder should be organized as follows:
```
data/
├── ADHOC/
│   ├── qrels/
│   ├── rankers/
├── ROBUST/
│   ├── robust03_qrels.txt
│   ├── rankers/
└── ROUTING/
│   ├── qrels/
│   ├── rankers/
```
where rankers is a directory of the rankers to fuse, qrels is the directory of the .txt files that are extracted from the qrels zip in ROUTING and ADHOC datasets 
### Note
The rankers data files are not uploaded to this GitHub repository as publishing them is not permitted. However, the QRELs file can be downloaded directly from the following link:

[robust03_qrels.txt](https://trec.nist.gov/data/qrels_eng/robust03_qrels.txt)

[qrels zip for ad hoc](https://trec.nist.gov/data/qrels_eng/qrels.151-200.201-250.disks1-3.all.tar.gz)

[qrels zip for routing](https://trec.nist.gov/data/qrels_eng/qrels.101-150.disk3.parts1-5.tar.gz)

## Steps to Run
1. **Generate full qrels .txt file for routing and ad hoc datasets:**
      Run the `creating_qrels_united_file.py` script to create the routing03_qrels.txt and adhoc03_qrels.txt files in their correct directories
   ```bash
   python "creating_qrels_united_file.py"
   ```
2. **Generate Per Query QRELs File:**
   Run the `per query qrels file.py` script to create QRELs text files for individual queries.
   ```bash
   python "per query qrels file.py"
   ```

3. **Create Folds:**
   Run the `create folds.py` script to create 5 folds (or more accurately cross-test folds) for cross-validation.
   ```bash
   python "create folds.py"
   ```

4. **Main File:**
   Use the main file to generate results.
   ```bash
   python main.py
   ```
