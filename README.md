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
```
### DVC Commands

```bash
dvc init
dvc repro
dvc metrics diff main
dvc metrics show
dvc add dataset.csv
dvc checkout ## to return the commit 
```
