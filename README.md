## `Churn Detection with use of CML & DVC Tools `
    * Using different approaches for solving imbalancing dataset.
    * Using different Algorithms also.
-------------------
### Note
> `cml-churn.yaml` file is attached to this directory. You can put it in `.github/workflows/cml-churn.yaml` as usual.
------------------------
### git commands
```bash
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/kzebibi/dvc-pipeline.git
git push --set-upstream origin main
git rm -r --cached 'dataset.csv'
git commit -m "stop tracking dataset.csv"
git checkout "commit hash"
<<<<<<< HEAD
git log --all
=======
>>>>>>> 65ee497ab73405621708ac887eb7df30f0e955ab
```
### DVC Commands

```bash
dvc init
dvc repro
dvc metrics diff main
dvc metrics show
<<<<<<< HEAD
dvc plots show roc_data.csv -x fpr -y tpr
dvc params diff 
dvc params diff 6b242e1
dvc metrics diff 6b242e
dvc add dataset.csv
dvc checkout ## to return the commit 

dvc dag
=======
dvc add dataset.csv
dvc checkout ## to return the commit 
```

```bash
git checkout HEAD~1 dataset.csv.dvc
dvc checkout
>>>>>>> 65ee497ab73405621708ac887eb7df30f0e955ab
```
## 
```bash
git checkout HEAD~1 dataset.csv.dvc
dvc checkout

```

```python
import yaml
yaml.safe_load(open("params.yaml"))
## or 
import dvc.api
dvc.api.params_show()

```


dvc remote remove myremote
dvc remote add --default myremote gdrive://1cNPh0ESwP2RTpM7oE59cbpzVQMnd_705
dvc remote modify myremote gdrive_client_id "K:/AI Track/12_MLOps/agoor/dvc pipline/google_drive_client_id.json.bak"
dvc remote modify myremote gdrive_use_service_account false