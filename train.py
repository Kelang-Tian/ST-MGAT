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
parser.add_argument('--data_path', type=str, default='data/METR-LA', help='data path')

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
parser.add_argument('--save', type=str, default='./experiment/stgat/', help='save path')
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
    continue_train = 0
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
    best_path = os.path.join(args.save, 'best_model.pkl')
    # / in path ？？？？

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
            # print("117:", trainx.shape)
            trainx = nn.functional.pad(trainx, (1, 0, 0, 0))    # ([64, 2, 207, 13])
            # print("119:", trainx.shape)
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
