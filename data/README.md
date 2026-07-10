# Data setup

## Sample dataset (current)

Source: https://www.kaggle.com/datasets/nih-chest-xrays/sample
5,606 chest X-rays, ~5 GB, subset of NIH ChestX-ray14.

Manual download only (Kaggle CLI blocked on this network) — download
the zip from the link above, extract it, and place files as:

```
data/sample/
├── images/              <- all PNG files directly here (ignore any nested
│                            sample/sample/images/ duplicate from the zip)
└── sample_labels.csv
```

Load it with:

```python
from data.dataset import ChestXray14Dataset

ds = ChestXray14Dataset(
    data_dir="data/sample",
    labels_csv="data/sample/sample_labels.csv",
)
```

## Full dataset (later, optional)

112,120 images, ~45 GB. Same Kaggle account works:
https://www.kaggle.com/datasets/nih-chest-xrays/data

Uses `Data_Entry_2017.csv` (default filename, no `labels_csv` override needed).
Not needed until the sample-trained pipeline works end to end.