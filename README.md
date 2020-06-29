#Spatial-Temporal Multi-Head Graph Attention Networks for Traffic Forecasting
<p align="center">
  <img width="600" height="450" src=./figure/architecture.jpg>
</p>

This is a Pytorch implementation of Spatial-Temporal Multi-Head Graph Attention Networks for Traffic Forecasting,
which combines the graph attention convolution (GAT) and the dilated convolution structure with gate mechanisms.
## Requirements
- see `requirements.txt`

## Data Preparation

1) Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).
`metr-la.h5` and `pems-bay.h5`should be put into the `data/` folder.
2)

```
# Create data directoriesdfrt yui 
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python data_preparation.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python data_preparation.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

## Run Demo
```
# change 'base_path' and 'best_model_path' to choose which dataset you want
python run_demo.py --base_path=./pre_train_model/BAY_dataset --best_model_path=stgat_1.45.pkl

# run pre trained models (stgcn, gwnet models about BAY datasets will be added)
python run_demo_baselines.py
```

### Folder structure
```
├── baselines
│   ├── experiment_base
│   ├── __init__.py
│   ├── gwnet.py
│   ├── rnn.py
│   ├── run_demo_baselines.py
│   ├── stgcn.py
│   └── train_base.csv
├── data
│   ├── METR-LA
│   ├── PEMS-BAY
│   ├── sensor_graph
│   ├── metr-la.h5
│   └── pems-bay.h5
├── experiment
├── pre_train_model
│   ├── BAY_dataset
│   └── LA_dataset
│
├── data_preparation.py
├── draw.py
├── model_stgat.py
├── README.md
├── requirements.txt
├── run_demo.py
├── train.py
└── util.py

```
