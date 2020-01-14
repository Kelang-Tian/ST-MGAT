import torch
import numpy as np
import argparse
import time
import util
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd


from fastprogress import progress_bar
import torch.nn.functional as F
from model_stgat import stgat

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='STGAT')
parser.add_argument('--adj_path', type=str, default='data/sensor_graph/adj_mx_distance_normalized.csv',
                    help='adj data path')
parser.add_argument('--data_path', type=str, default='data/METR-LA12_shuffle', help='data path')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--num_layers', type=int, default=1, help='layers of gat')
parser.add_argument('--in_dim', type=int, default=2, help='number of nodes features')
parser.add_argument('--num_hidden', type=int, default=8, help='number of hidden in gat')
parser.add_argument('--out_dim', type=int, default=8, help='number of out_dim')
parser.add_argument('--heads', type=int, default=8, help='number of out_dim')
parser.add_argument('--feat_drop', type=int, default=0.6, help='  ')
parser.add_argument('--attn_drop', type=int, default=0.6, help='  ')
parser.add_argument('--negative_slope', type=int, default=0.2, help='  ')
parser.add_argument('--activation', action="store_true", default=F.elu, help='  ')
parser.add_argument('--residual', action="store_true", default=False, help='  ')
parser.add_argument('--interval', type=int, default=100, help='')
parser.add_argument('--num_epochs', type=int, default=100, help='')
parser.add_argument('--save', type=str, default='./experiment/combine/continue_train_la_shuffle2/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--seq_len', type=int, default=12, help='time length of inputs')
parser.add_argument('--pre_len', type=int, default=12, help='time length of prediction')

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Let's use {} GPU!".format(device))
else:
    device = torch.device("cpu")


def evaluate_all(pred, target):
    mape = util.masked_mape(pred, target, 0.0).item()
    rmse = util.masked_rmse(pred, target, 0.0).item()
    mae = util.masked_mae(pred, target, 0.0).item()
    return mape, rmse, mae


def run_demo(best_path, record_save_path):
    print("============Begin Testing============")
    test_record_path = f'{record_save_path}/stgat_test_record.csv'
    dataloader = util.load_dataset(device, args.data_path, args.batch_size, args.batch_size, args.batch_size)
    g_temp = util.add_nodes_edges(adj_filename=args.adj_path, num_of_vertices=args.num_nodes)
    scaler = dataloader['scaler']
    run_gconv = 1
    lr_decay_rate = 0.97

    model = stgat(g=g_temp, run_gconv=run_gconv)
    model.to(device)
    model.zero_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

    model.load_state_dict(torch.load(best_path))

    outputs = []
    target = torch.Tensor(dataloader['y_test']).to(device)
    target = target[:, :, :, 0]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        testx = nn.functional.pad(testx, (1, 0, 0, 0))
        with torch.no_grad():
            pred = model.forward(testx).squeeze(3)
        outputs.append(pred)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:target.size(0), ...]
    test_record, amape, armse, amae = [], [], [], []

    pred = scaler.inverse_transform(yhat)
    for i in range(12):
        pred_t = pred[:, i, :]
        real_target = target[:, i, :]
        evaluation = evaluate_all(pred_t, real_target)
        log = 'test for horizon {:d}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test MAE: {:.4f}'
        print(log.format(i + 1, evaluation[0], evaluation[1], evaluation[2]))
        amape.append(evaluation[0])
        armse.append(evaluation[1])
        amae.append(evaluation[2])
        test_record.append([x for x in evaluation])
    test_record_df = pd.DataFrame(test_record, columns=['mape', 'rmse', 'mae']).rename_axis('t')
    test_record_df.round(3).to_csv(test_record_path)
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    print("=" * 10)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  New Folder: ", path)
    else:
        print("---  Folder already exists:", path)


if __name__ == "__main__":
    base_path = './pre_train_model/BAY_dataset'     # or ./pre_train_model/LA_dataset/
    best_path = f'{base_path}/stgat_1.45.pkl'
    # best_path = './pre_train_model/BAY_dataset/stgat_2.88.pkl'

    record_save_path = f'{base_path}/stgat'
    mkdir(record_save_path)

    run_demo(best_path, record_save_path)
