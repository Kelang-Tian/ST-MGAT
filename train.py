import torch
import numpy as np
import argparse
import time
import util
import os
import matplotlib.pyplot as plt
import torch.nn as nn


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


def evaluate(pred, target):
    mape = util.masked_mape(pred, target, 0.0).item()
    rmse = util.masked_rmse(pred, target, 0.0).item()
    return mape, rmse

def evaluate_all(pred, target):
    mape = util.masked_mape(pred, target, 0.0).item()
    rmse = util.masked_rmse(pred, target, 0.0).item()
    mae = util.masked_mae(pred, target, 0.0).item()
    return mape, rmse, mae

def main():
    print("*" * 10)
    print(args)
    print("*" * 10)

    dataloader = util.load_dataset(device, args.data_path, args.batch_size, args.batch_size, args.batch_size)
    g_temp = util.add_nodes_edges(adj_filename=args.adj_path, num_of_vertices=args.num_nodes)

    scaler = dataloader['scaler']
    test_scaler = dataloader['test_scaler']

    clip = 3
    run_gconv = 1
    best_mae = 100
    continue_train = 1
    lr_decay_rate = 0.97
    record = []

    model = stgat(g=g_temp, run_gconv=run_gconv)
    model.to(device)
    model.zero_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

    loss = util.masked_mae
    best_path = os.path.join(args.save, 'best_model.pth')
    # 此处存疑，不知是否应该加/在路径中？？？？

    # if you want to train model with the last parameters
    if continue_train:
        path = './experiment/last_record.pkl'
        print("reload the model from :{}", path)
        model.load_state_dict(torch.load(path))

    print("============Begin Training============")
    his_loss, val_time, train_time = [], [], []
    for epoch in range(args.num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        train_loss, train_mape, train_rmse = [], [], []
        t1 = time.time()
        t = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)  # x: (64, 2, 207, 12)
            trainy = torch.Tensor(y).to(device)     # (64, 12, 207, 2)
            trainy = trainy[:, :, :, 0]             # only predict speed/ you can replace it with any features you want

            if trainx.shape[0] != args.batch_size:
                continue

            trainx = nn.functional.pad(trainx, (1, 0, 0, 0))
            pred = model.forward(trainx).squeeze(3)
            pred = scaler.inverse_transform(pred)
            # pred = test_scaler.inverse_transform(pred)

            if iter == 0:
                print("trainy:", trainy.shape)      # ([64, 12, 207])
                print("pred:", pred.shape)

            # loss_train = loss_MSE(pred, trainy)
            mae_loss_train = loss(pred, trainy, 0.0)
            optimizer.zero_grad()
            mae_loss_train.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            evaluation = evaluate(pred, trainy)
            train_loss.append(mae_loss_train.item())
            train_mape.append(evaluation[0])
            train_rmse.append(evaluation[1])

            if iter % args.interval == 0:
                log = 'Iter: {:03d}|Train Loss: {:.4f}|Time: {:.4f}'
                print(log.format(iter, train_loss[-1], time.time() - t), flush=True)
                t = time.time()
        scheduler.step()
        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss, valid_mape, valid_rmse = [], [], []
        s1 = time.time()
        for iter, (x_val, y_val) in enumerate(dataloader['val_loader'].get_iterator()):
            inputs_val = torch.Tensor(x_val).to(device).transpose(1, 3)  # x: (64, 24, 207, 2)
            labels_val = torch.Tensor(y_val).to(device)
            labels_val = labels_val[:, :, :, 0]

            inputs_val = nn.functional.pad(inputs_val, (1, 0, 0, 0))
            pred_val = model.forward(inputs_val).squeeze(3)
            pred_val = scaler.inverse_transform(pred_val)
            # pred_val = test_scaler.inverse_transform(pred_val)

            mae_loss_val = loss(pred_val, labels_val, 0.0)
            optimizer.zero_grad()
            mae_loss_val.backward()
            evaluation = evaluate(pred_val, labels_val)

            valid_loss.append(mae_loss_val.item())
            valid_mape.append(evaluation[0])
            valid_rmse.append(evaluation[1])

        s2 = time.time()
        # log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        # print(log.format(epoch, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        # save best model parameters, evaluation of every epochs
        message = dict(train_loss=mtrain_loss, train_mape=mtrain_mape, train_rmse=mtrain_rmse,
                       valid_loss=mvalid_loss, valid_mape=mvalid_mape, valid_rmse=mvalid_rmse)
        message = pd.Series(message)
        record.append(message)
        record_df = pd.DataFrame(record)
        record_df.round(3).to_csv(f'{args.save}/record.csv')

        if message.valid_loss < best_mae:
            torch.save(model.state_dict(), best_path)
            best_mae = message.valid_loss
            epochs_since_best_mae = 0
            best_epoch = epoch
        else:
            epochs_since_best_mae += 1

        log = 'Epoch: {:03d}, Training Time: {:.4f}/epoch,\n' \
              'Train Loss: {:.4f}  \n' \
              'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}' \
              'best epoch: {} , best val_loss: {}, epochs since best: {}'
        print(log.format(epoch, (t2 - t1),
                         mtrain_loss,
                         mvalid_loss, mvalid_mape, mvalid_rmse,
                         best_epoch, record_df.valid_loss.min().round(3), epochs_since_best_mae),
              flush=True)
        print("#" * 20)

    # finished train
    print("=" * 10)
    print("Average Train Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Valid Time: {:.4f} secs".format(np.mean(val_time)))
    print("=" * 10)

    # Testing
    bestid = np.argmin(his_loss)
    print("bestid: ", bestid)
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
    test_record_df.round(3).to_csv(f'{args.save}/test_record.csv')
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
    mkdir(args.save)
    main()


"""
bestid:  98
path:  ./experiment/combine/shuffle_batched_gat_2l_lr_decay/epoch98_2.97.pkl
201 y_test: torch.Size([6850, 12, 207])
==========
yhat: torch.Size([6850, 12, 207])
target: torch.Size([6850, 12, 207])
test data for 1, Test MAPE: 5.2449, Test RMSE: 3.8783, Test MAE: 2.2221
test data for 2, Test MAPE: 6.2688, Test RMSE: 4.7217, Test MAE: 2.5051
test data for 3, Test MAPE: 6.9492, Test RMSE: 5.2209, Test MAE: 2.6865
test data for 4, Test MAPE: 7.4869, Test RMSE: 5.5772, Test MAE: 2.8172
test data for 5, Test MAPE: 7.9169, Test RMSE: 5.8529, Test MAE: 2.9330
test data for 6, Test MAPE: 8.3146, Test RMSE: 6.0767, Test MAE: 3.0239
test data for 7, Test MAPE: 8.6419, Test RMSE: 6.2648, Test MAE: 3.1102
test data for 8, Test MAPE: 8.9286, Test RMSE: 6.4290, Test MAE: 3.1845
test data for 9, Test MAPE: 9.1654, Test RMSE: 6.5784, Test MAE: 3.2558
test data for 10, Test MAPE: 9.4073, Test RMSE: 6.7209, Test MAE: 3.3114
test data for 11, Test MAPE: 9.7494, Test RMSE: 6.8822, Test MAE: 3.3878
test data for 12, Test MAPE: 9.9665, Test RMSE: 7.0211, Test MAE: 3.4582
On average over 12 horizons, Test MAE: 2.9913, Test MAPE: 8.1700, Test RMSE: 5.9353

bestid:  145
path:  ./experiment/combine/continue_train_la_shuffle/epoch145_2.88.pkl
201 y_test: torch.Size([6850, 12, 207])
==========
yhat: torch.Size([6850, 12, 207])
target: torch.Size([6850, 12, 207])
test data for 1, Test MAPE: 5.1872, Test RMSE: 3.8578, Test MAE: 2.2097
test data for 2, Test MAPE: 6.1635, Test RMSE: 4.6639, Test MAE: 2.4798
test data for 3, Test MAPE: 6.7759, Test RMSE: 5.1203, Test MAE: 2.6453
test data for 4, Test MAPE: 7.2431, Test RMSE: 5.4244, Test MAE: 2.7544
test data for 5, Test MAPE: 7.5912, Test RMSE: 5.6581, Test MAE: 2.8518
test data for 6, Test MAPE: 7.9169, Test RMSE: 5.8512, Test MAE: 2.9257
test data for 7, Test MAPE: 8.1781, Test RMSE: 6.0058, Test MAE: 2.9963
test data for 8, Test MAPE: 8.4093, Test RMSE: 6.1480, Test MAE: 3.0583
test data for 9, Test MAPE: 8.5927, Test RMSE: 6.2825, Test MAE: 3.1191
test data for 10, Test MAPE: 8.8178, Test RMSE: 6.4177, Test MAE: 3.1701
test data for 11, Test MAPE: 9.1277, Test RMSE: 6.5743, Test MAE: 3.2407
test data for 12, Test MAPE: 9.3537, Test RMSE: 6.7198, Test MAE: 3.3107
On average over 12 horizons, Test MAE: 2.8968, Test MAPE: 7.7798, Test RMSE: 5.7270
"""


"""
bestid:  99
path:  ./experiment/combine/bay_shuffle_batched_gat_2l_lr_decay/epoch99_1.51.pkl
201 y_test: torch.Size([10419, 12, 325])
==========
yhat: torch.Size([10419, 12, 325])
target: torch.Size([10419, 12, 325])
test data for 1, Test MAPE: 1.5964, Test RMSE: 1.5188, Test MAE: 0.8356
test data for 2, Test MAPE: 2.1939, Test RMSE: 2.1716, Test MAE: 1.0977
test data for 3, Test MAPE: 2.6323, Test RMSE: 2.6609, Test MAE: 1.2738
test data for 4, Test MAPE: 2.9648, Test RMSE: 3.0165, Test MAE: 1.3973
test data for 5, Test MAPE: 3.2209, Test RMSE: 3.2762, Test MAE: 1.4882
test data for 6, Test MAPE: 3.4300, Test RMSE: 3.4747, Test MAE: 1.5616
test data for 7, Test MAPE: 3.6045, Test RMSE: 3.6270, Test MAE: 1.6209
test data for 8, Test MAPE: 3.7476, Test RMSE: 3.7500, Test MAE: 1.6712
test data for 9, Test MAPE: 3.8710, Test RMSE: 3.8539, Test MAE: 1.7162
test data for 10, Test MAPE: 3.9993, Test RMSE: 3.9561, Test MAE: 1.7617
test data for 11, Test MAPE: 4.1172, Test RMSE: 4.0480, Test MAE: 1.8034
test data for 12, Test MAPE: 4.2424, Test RMSE: 4.1451, Test MAE: 1.8502
On average over 12 horizons, Test MAE: 1.5065, Test MAPE: 3.3017, Test RMSE: 3.2916
"""

"""
bestid:  99
path:  ./experiment/combine/continue_train_bay_shuffle/epoch99_1.47.pkl
201 y_test: torch.Size([10419, 12, 325])
==========
yhat: torch.Size([10419, 12, 325])
target: torch.Size([10419, 12, 325])
test data for 1, Test MAPE: 1.5865, Test RMSE: 1.5130, Test MAE: 0.8315
test data for 2, Test MAPE: 2.1732, Test RMSE: 2.1537, Test MAE: 1.0891
test data for 3, Test MAPE: 2.5966, Test RMSE: 2.6265, Test MAE: 1.2586
test data for 4, Test MAPE: 2.9073, Test RMSE: 2.9620, Test MAE: 1.3740
test data for 5, Test MAPE: 3.1442, Test RMSE: 3.2018, Test MAE: 1.4575
test data for 6, Test MAPE: 3.3370, Test RMSE: 3.3844, Test MAE: 1.5249
test data for 7, Test MAPE: 3.4961, Test RMSE: 3.5238, Test MAE: 1.5782
test data for 8, Test MAPE: 3.6293, Test RMSE: 3.6360, Test MAE: 1.6243
test data for 9, Test MAPE: 3.7471, Test RMSE: 3.7357, Test MAE: 1.6661
test data for 10, Test MAPE: 3.8636, Test RMSE: 3.8359, Test MAE: 1.7090
test data for 11, Test MAPE: 3.9775, Test RMSE: 3.9286, Test MAE: 1.7494
test data for 12, Test MAPE: 4.1082, Test RMSE: 4.0323, Test MAE: 1.7977
On average over 12 horizons, Test MAE: 1.4717, Test MAPE: 3.2139, Test RMSE: 3.2111



bestid:  96
path:  ./experiment/combine/continue_continue_train_bay_shuffle/epoch96_1.45.pkl
201 y_test: torch.Size([10419, 12, 325])
==========
yhat: torch.Size([10419, 12, 325])
target: torch.Size([10419, 12, 325])
test data for 1, Test MAPE: 1.5854, Test RMSE: 1.5103, Test MAE: 0.8305
test data for 2, Test MAPE: 2.1630, Test RMSE: 2.1441, Test MAE: 1.0855
test data for 3, Test MAPE: 2.5722, Test RMSE: 2.6067, Test MAE: 1.2513
test data for 4, Test MAPE: 2.8688, Test RMSE: 2.9322, Test MAE: 1.3627
test data for 5, Test MAPE: 3.0928, Test RMSE: 3.1616, Test MAE: 1.4428
test data for 6, Test MAPE: 3.2751, Test RMSE: 3.3346, Test MAE: 1.5068
test data for 7, Test MAPE: 3.4230, Test RMSE: 3.4667, Test MAE: 1.5580
test data for 8, Test MAPE: 3.5503, Test RMSE: 3.5727, Test MAE: 1.6024
test data for 9, Test MAPE: 3.6618, Test RMSE: 3.6692, Test MAE: 1.6425
test data for 10, Test MAPE: 3.7781, Test RMSE: 3.7691, Test MAE: 1.6846
test data for 11, Test MAPE: 3.8927, Test RMSE: 3.8640, Test MAE: 1.7250
test data for 12, Test MAPE: 4.0279, Test RMSE: 3.9713, Test MAE: 1.7741
On average over 12 horizons, Test MAE: 1.4555, Test MAPE: 3.1576, Test RMSE: 3.1669
"""


"""
bestid:  40
path:  ./experiment/combine/no_gconv_LA12_shuffl/epoch40_3.41.pkl
201 y_test: torch.Size([6850, 12, 207])
==========
yhat: torch.Size([6850, 12, 207])
target: torch.Size([6850, 12, 207])
test data for 1, Test MAPE: 5.4220, Test RMSE: 4.0468, Test MAE: 2.2941
test data for 2, Test MAPE: 6.6058, Test RMSE: 5.0556, Test MAE: 2.6334
test data for 3, Test MAPE: 7.4530, Test RMSE: 5.7208, Test MAE: 2.8814
test data for 4, Test MAPE: 8.2228, Test RMSE: 6.2450, Test MAE: 3.0833
test data for 5, Test MAPE: 8.8856, Test RMSE: 6.6767, Test MAE: 3.2745
test data for 6, Test MAPE: 9.5480, Test RMSE: 7.0517, Test MAE: 3.4416
test data for 7, Test MAPE: 10.1312, Test RMSE: 7.3614, Test MAE: 3.5951
test data for 8, Test MAPE: 10.6934, Test RMSE: 7.6490, Test MAE: 3.7387
test data for 9, Test MAPE: 11.1815, Test RMSE: 7.9042, Test MAE: 3.8746
test data for 10, Test MAPE: 11.6734, Test RMSE: 8.1357, Test MAE: 3.9884
test data for 11, Test MAPE: 12.2398, Test RMSE: 8.3625, Test MAE: 4.1132
test data for 12, Test MAPE: 12.6592, Test RMSE: 8.5570, Test MAE: 4.2245
On average over 12 horizons, Test MAE: 3.4286, Test MAPE: 9.5596, Test RMSE: 6.8972
"""

"""
bestid:  47
path:  ./experiment/combine/no_gconv_BAY12_shuffl/epoch47_1.79.pkl
201 y_test: torch.Size([10419, 12, 325])
==========
yhat: torch.Size([10419, 12, 325])
target: torch.Size([10419, 12, 325])
test data for 1, Test MAPE: 1.6327, Test RMSE: 1.5748, Test MAE: 0.8590
test data for 2, Test MAPE: 2.2952, Test RMSE: 2.3264, Test MAE: 1.1536
test data for 3, Test MAPE: 2.8362, Test RMSE: 2.9379, Test MAE: 1.3725
test data for 4, Test MAPE: 3.3030, Test RMSE: 3.4261, Test MAE: 1.5459
test data for 5, Test MAPE: 3.7196, Test RMSE: 3.8237, Test MAE: 1.6919
test data for 6, Test MAPE: 4.0937, Test RMSE: 4.1586, Test MAE: 1.8196
test data for 7, Test MAPE: 4.4263, Test RMSE: 4.4400, Test MAE: 1.9298
test data for 8, Test MAPE: 4.7229, Test RMSE: 4.6777, Test MAE: 2.0278
test data for 9, Test MAPE: 4.9931, Test RMSE: 4.8817, Test MAE: 2.1167
test data for 10, Test MAPE: 5.2583, Test RMSE: 5.0679, Test MAE: 2.2014
test data for 11, Test MAPE: 5.4838, Test RMSE: 5.2285, Test MAE: 2.2759
test data for 12, Test MAPE: 5.7120, Test RMSE: 5.3822, Test MAE: 2.3504
On average over 12 horizons, Test MAE: 1.7787, Test MAPE: 4.0397, Test RMSE: 3.9938
"""
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
from STGAT import STGAT, STGAT_2
from Tattention_STGAT import Tattention_STGAT
from notime_series_STGAT import notime_STGAT
from stgat_train import Trainer
from no_batch_STGAT import gat_mean_STGAT, gat_mean_STGAT_2
# from pred_STGAT import pred3_STGAT
from combine_gw_gat import GAT_GW_Net
from batch_g_gat import batch_g_gat


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
parser.add_argument('--save', type=str, default='./experiment/draw_LA/no_gconv_STGAT_LA', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--seq_len', type=int, default=12, help='time length of inputs')
parser.add_argument('--pre_len', type=int, default=12, help='time length of prediction')

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Let's use {} GPU!".format(device))
else:
    device = torch.device("cpu")


def evaluate(pred, target):
    mape = util.masked_mape(pred, target, 0.0).item()
    rmse = util.masked_rmse(pred, target, 0.0).item()
    # mae = util.masked_mae(pred, target, 0.0).item()
    return mape, rmse


def evaluate_all(pred, target):
    mape = util.masked_mape(pred, target, 0.0).item()
    rmse = util.masked_rmse(pred, target, 0.0).item()
    mae = util.masked_mae(pred, target, 0.0).item()
    return mape, rmse, mae

def main():
    print("*" * 10)
    print(args)
    print("*" * 10)
    dataloader = util.load_dataset(device, args.data_path, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    test_scaler = dataloader['test_scaler']
    g_temp = util.add_nodes_edges(adj_filename=args.adj_path, num_of_vertices=args.num_nodes)
    record = []

    clip = 3
    best_mae = 100
    lr_decay_rate = 0.97
    run_gconv = 0
    loss = util.masked_mae
    model = batch_g_gat(g_temp, run_gconv)
    print("run_gconv:", run_gconv)
    model.to(device)
    model.zero_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)
    bar = progress_bar(list(range(1, args.num_epochs + 1)))
    best_path = os.path.join(args.save, 'best_model.pth')
    his_loss, val_time, train_time = [], [], []
    print("============Begin Training============")
    for epoch in bar:
        # print('-' * 10)
        # print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        train_loss, train_mape, train_rmse = [], [], []
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)  # x: (64, 2, 207, 12)
            trainy = torch.Tensor(y).to(device)     # (64, 12, 207, 2)
            trainy = trainy[:, :, :, 0]

            if trainx.shape[0] != args.batch_size:
                continue

            trainx = nn.functional.pad(trainx, (1, 0, 0, 0))
            pred = model.forward(trainx).squeeze(3)
            pred = scaler.inverse_transform(pred)
            # pred = test_scaler.inverse_transform(pred)
            # loss_train = loss_MSE(pred, trainy)
            mae_loss_train = loss(pred, trainy, 0.0)
            optimizer.zero_grad()
            mae_loss_train.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            evaluation = evaluate(pred, trainy)
            train_loss.append(mae_loss_train.item())
            train_mape.append(evaluation[0])
            train_rmse.append(evaluation[1])
        scheduler.step()

        # validation
        valid_loss, valid_mape, valid_rmse = [], [], []
        for iter, (x_val, y_val) in enumerate(dataloader['val_loader'].get_iterator()):
            inputs_val = torch.Tensor(x_val).to(device).transpose(1, 3)  # x: (64, 24, 207, 2)
            labels_val = torch.Tensor(y_val).to(device)
            labels_val = labels_val[:, :, :, 0]

            inputs_val = nn.functional.pad(inputs_val, (1, 0, 0, 0))
            pred_val = model.forward(inputs_val).squeeze(3)
            pred_val = scaler.inverse_transform(pred_val)
            # pred_val = test_scaler.inverse_transform(pred_val)

            mae_loss_val = loss(pred_val, labels_val, 0.0)
            optimizer.zero_grad()
            mae_loss_val.backward()
            evaluation = evaluate(pred_val, labels_val)

            valid_loss.append(mae_loss_val.item())
            valid_mape.append(evaluation[0])
            valid_rmse.append(evaluation[1])

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        message = dict(train_loss=mtrain_loss, train_mape=mtrain_mape, train_rmse=mtrain_rmse,
                       valid_loss=mvalid_loss, valid_mape=mvalid_mape, valid_rmse=mvalid_rmse)
        message = pd.Series(message)
        record.append(message)

        # save model parameters
        if message.valid_loss < best_mae:
            torch.save(model.state_dict(), best_path)
            best_mae = message.valid_loss
            epochs_since_best_mae = 0
            best_epoch = epoch
        else:
            epochs_since_best_mae += 1

        record_df = pd.DataFrame(record)
        bar.comment = f'best epoch: {best_epoch}, best val_loss: {record_df.valid_loss.min(): .3f}, current val_loss: {message.valid_loss:.3f},current train loss: {message.train_loss: .3f}'
        record_df.round(3).to_csv(f'{args.save}/record.csv')

    # Testing
    bestid = np.argmin(his_loss)
    # print("bestid: ", bestid)
    # print("best_epoch: ", best_epoch)
    # print("best_path: ", best_path)
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
    amape, armse, amae = [], [], []
    # test_mape, test_rmse, test_mae = evaluate_all(yhat, target)
    # print("=" * 10)
    # print("yhat:", yhat.shape)  # yhat: torch.Size([6850, 12, 207])
    # print("target:", target.shape)  # target: torch.Size([6850, 12, 207])

    pred = scaler.inverse_transform(yhat)
    # pred = test_scaler.inverse_transform(yhat)
    test_record = []
    for i in range(12):
        pred_t = pred[:, i, :]
        real_target = target[:, i, :]
        evaluation = evaluate_all(pred_t, real_target)
        log = 'test data for {:d}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test MAE: {:.4f}'
        print(log.format(i + 1, evaluation[0], evaluation[1], evaluation[2]))
        amape.append(evaluation[0])
        armse.append(evaluation[1])
        amae.append(evaluation[2])
        test_record.append([x for x in evaluation])
    test_record_df = pd.DataFrame(test_record, columns=['mape', 'rmse', 'mae']).rename_axis('t')
    test_record_df.round(3).to_csv(f'{args.save}/test_record.csv')
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder  ---", path)
    else:
        print("---  There is this folder!  ---", path)


if __name__ == "__main__":
    mkdir(args.save)
    main()


"""
bestid:  98
path:  ./experiment/combine/shuffle_batched_gat_2l_lr_decay/epoch98_2.97.pkl
201 y_test: torch.Size([6850, 12, 207])
==========
yhat: torch.Size([6850, 12, 207])
target: torch.Size([6850, 12, 207])
test data for 1, Test MAPE: 5.2449, Test RMSE: 3.8783, Test MAE: 2.2221
test data for 2, Test MAPE: 6.2688, Test RMSE: 4.7217, Test MAE: 2.5051
test data for 3, Test MAPE: 6.9492, Test RMSE: 5.2209, Test MAE: 2.6865
test data for 4, Test MAPE: 7.4869, Test RMSE: 5.5772, Test MAE: 2.8172
test data for 5, Test MAPE: 7.9169, Test RMSE: 5.8529, Test MAE: 2.9330
test data for 6, Test MAPE: 8.3146, Test RMSE: 6.0767, Test MAE: 3.0239
test data for 7, Test MAPE: 8.6419, Test RMSE: 6.2648, Test MAE: 3.1102
test data for 8, Test MAPE: 8.9286, Test RMSE: 6.4290, Test MAE: 3.1845
test data for 9, Test MAPE: 9.1654, Test RMSE: 6.5784, Test MAE: 3.2558
test data for 10, Test MAPE: 9.4073, Test RMSE: 6.7209, Test MAE: 3.3114
test data for 11, Test MAPE: 9.7494, Test RMSE: 6.8822, Test MAE: 3.3878
test data for 12, Test MAPE: 9.9665, Test RMSE: 7.0211, Test MAE: 3.4582
On average over 12 horizons, Test MAE: 2.9913, Test MAPE: 8.1700, Test RMSE: 5.9353
"""


"""
bestid:  99
path:  ./experiment/combine/bay_shuffle_batched_gat_2l_lr_decay/epoch99_1.51.pkl
201 y_test: torch.Size([10419, 12, 325])
==========
yhat: torch.Size([10419, 12, 325])
target: torch.Size([10419, 12, 325])
test data for 1, Test MAPE: 1.5964, Test RMSE: 1.5188, Test MAE: 0.8356
test data for 2, Test MAPE: 2.1939, Test RMSE: 2.1716, Test MAE: 1.0977
test data for 3, Test MAPE: 2.6323, Test RMSE: 2.6609, Test MAE: 1.2738
test data for 4, Test MAPE: 2.9648, Test RMSE: 3.0165, Test MAE: 1.3973
test data for 5, Test MAPE: 3.2209, Test RMSE: 3.2762, Test MAE: 1.4882
test data for 6, Test MAPE: 3.4300, Test RMSE: 3.4747, Test MAE: 1.5616
test data for 7, Test MAPE: 3.6045, Test RMSE: 3.6270, Test MAE: 1.6209
test data for 8, Test MAPE: 3.7476, Test RMSE: 3.7500, Test MAE: 1.6712
test data for 9, Test MAPE: 3.8710, Test RMSE: 3.8539, Test MAE: 1.7162
test data for 10, Test MAPE: 3.9993, Test RMSE: 3.9561, Test MAE: 1.7617
test data for 11, Test MAPE: 4.1172, Test RMSE: 4.0480, Test MAE: 1.8034
test data for 12, Test MAPE: 4.2424, Test RMSE: 4.1451, Test MAE: 1.8502
On average over 12 horizons, Test MAE: 1.5065, Test MAPE: 3.3017, Test RMSE: 3.2916
"""

"""
bestid:  40
path:  ./experiment/combine/no_gconv_LA12_shuffl/epoch40_3.41.pkl
201 y_test: torch.Size([6850, 12, 207])
==========
yhat: torch.Size([6850, 12, 207])
target: torch.Size([6850, 12, 207])
test data for 1, Test MAPE: 5.4220, Test RMSE: 4.0468, Test MAE: 2.2941
test data for 2, Test MAPE: 6.6058, Test RMSE: 5.0556, Test MAE: 2.6334
test data for 3, Test MAPE: 7.4530, Test RMSE: 5.7208, Test MAE: 2.8814
test data for 4, Test MAPE: 8.2228, Test RMSE: 6.2450, Test MAE: 3.0833
test data for 5, Test MAPE: 8.8856, Test RMSE: 6.6767, Test MAE: 3.2745
test data for 6, Test MAPE: 9.5480, Test RMSE: 7.0517, Test MAE: 3.4416
test data for 7, Test MAPE: 10.1312, Test RMSE: 7.3614, Test MAE: 3.5951
test data for 8, Test MAPE: 10.6934, Test RMSE: 7.6490, Test MAE: 3.7387
test data for 9, Test MAPE: 11.1815, Test RMSE: 7.9042, Test MAE: 3.8746
test data for 10, Test MAPE: 11.6734, Test RMSE: 8.1357, Test MAE: 3.9884
test data for 11, Test MAPE: 12.2398, Test RMSE: 8.3625, Test MAE: 4.1132
test data for 12, Test MAPE: 12.6592, Test RMSE: 8.5570, Test MAE: 4.2245
On average over 12 horizons, Test MAE: 3.4286, Test MAPE: 9.5596, Test RMSE: 6.8972
"""

"""
bestid:  47
path:  ./experiment/combine/no_gconv_BAY12_shuffl/epoch47_1.79.pkl
201 y_test: torch.Size([10419, 12, 325])
==========
yhat: torch.Size([10419, 12, 325])
target: torch.Size([10419, 12, 325])
test data for 1, Test MAPE: 1.6327, Test RMSE: 1.5748, Test MAE: 0.8590
test data for 2, Test MAPE: 2.2952, Test RMSE: 2.3264, Test MAE: 1.1536
test data for 3, Test MAPE: 2.8362, Test RMSE: 2.9379, Test MAE: 1.3725
test data for 4, Test MAPE: 3.3030, Test RMSE: 3.4261, Test MAE: 1.5459
test data for 5, Test MAPE: 3.7196, Test RMSE: 3.8237, Test MAE: 1.6919
test data for 6, Test MAPE: 4.0937, Test RMSE: 4.1586, Test MAE: 1.8196
test data for 7, Test MAPE: 4.4263, Test RMSE: 4.4400, Test MAE: 1.9298
test data for 8, Test MAPE: 4.7229, Test RMSE: 4.6777, Test MAE: 2.0278
test data for 9, Test MAPE: 4.9931, Test RMSE: 4.8817, Test MAE: 2.1167
test data for 10, Test MAPE: 5.2583, Test RMSE: 5.0679, Test MAE: 2.2014
test data for 11, Test MAPE: 5.4838, Test RMSE: 5.2285, Test MAE: 2.2759
test data for 12, Test MAPE: 5.7120, Test RMSE: 5.3822, Test MAE: 2.3504
On average over 12 horizons, Test MAE: 1.7787, Test MAPE: 4.0397, Test RMSE: 3.9938
"""


