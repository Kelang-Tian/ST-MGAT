import torch
import numpy as np
import argparse
import time
import os
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import sys
sys.path.append("..")
import util
from baselines.gwnet import GWNET
from baselines.stgcn import STGCN
from baselines.rnn import LSTM


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('--adj_path', type=str, default='../data/sensor_graph/adj_mx.pkl',
                    help='adj data path')
parser.add_argument('--data_path', type=str, default='../data/METR-LA12', help='data path')

parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')


parser.add_argument('--in_dim', type=int, default=2, help='number of nodes features')
parser.add_argument('--num_layers', type=int, default=1, help='layers of gat')
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

# parser.add_argument('--save', type=str, default='../experiment/stgcn/LA12/', help='save path')
parser.add_argument('--save', type=str, default='./experiment_base/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
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
    mae = util.masked_mae(pred, target, 0.0).item()
    return mape, rmse, mae


def main():
    print("*" * 10)
    print(args)
    print("*" * 10)
    dataloader = util.load_dataset(device, args.data_path, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    print("scaler: ", scaler)
    model_type = "GWaveNet"    # HA / SVR / ARIMA / STGCN / GWaveNet / LSTM

    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adj_path, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    _, _, A = util.load_pickle(args.adj_path)
    A_wave = util.get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).to(device)
    # print("A_wave:", A_wave.shape, type(A_wave))
    best_path = os.path.join(args.save, 'best_model.pth')
    best_mae = 100

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]
    if args.aptonly:
        supports = None

    if model_type == "GWaveNet":
        print("=========Model:GWaveNet=========")
        print("with scaler")
        model = GWNET(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
                      addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length,
                      residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid * 8,
                      end_channels=args.nhid * 16)

    if model_type == "STGCN":
        print("=========Model:STGCN=========")
        print("with scaler")
        model = STGCN(A_wave.shape[0], 2, num_timesteps_input=12, num_timesteps_output=12)

    if model_type == "LSTM":
        print("=========Model:LSTM=========")
        input_dim = 2
        hidden_dim = 2
        output_dim = 2
        model = LSTM(input_dim, hidden_dim, output_dim)

    best_path = f'{args.save}/{model_type}.pkl'
    record = []
    model.to(device)
    model.zero_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss_MSE = torch.nn.MSELoss()
    loss_gwnet = util.masked_mae
    loss_stgcn = util.masked_mae

    print("============Begin Training============")
    his_loss = []
    val_time = []
    train_time = []
    for epoch in range(args.num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, args.num_epochs))
        train_loss, train_mape, train_rmse, train_mae = [], [], [], []
        t1 = time.time()
        t = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)  # x: (64, 24, 207, 2)
            trainy = torch.Tensor(y).to(device)  # y: (64, 12, 207, 2)
            if trainx.shape[0] != args.batch_size:
                continue

            if model_type == "GWaveNet":
                trainx = trainx.transpose(1, 3)
                trainy = trainy.transpose(1, 3)
                trainy = trainy[:, 0, :, :]
                trainy = torch.unsqueeze(trainy, dim=1)
                trainx = nn.functional.pad(trainx, (1, 0, 0, 0))

                pred = model.forward(trainx)
                pred = pred.transpose(1, 3)
                pred = scaler.inverse_transform(pred)
                loss_train = loss_gwnet(pred, trainy, 0.0)

            if model_type == "STGCN":
                # (batch_size,num_timesteps,num_nodes,num_features=in_channels)
                # ->(batch_size,num_nodes,num_timesteps,num_features=in_channels)
                trainx = trainx.permute(0, 2, 1, 3)
                trainy = trainy[:, :, :, 0].permute(0, 2, 1)
                pred = model(A_wave, trainx)
                # pred = scaler.inverse_transform(pred)
                # loss_train = loss_MSE(pred, trainy)
                loss_train = loss_stgcn(pred, trainy, 0.0)

            if model_type == "rnn":
                [batch_size, step_size, num_of_vertices, fea_size] = trainx.size()
                trainx = trainx.permute(0, 2, 1, 3)
                trainx = trainx.reshape(-1, step_size, fea_size)
                trainy = trainy.reshape(-1, 1, fea_size)
                trainy = trainy[:, 0, :]
                pred = model.loop(trainx)
                loss_train = loss_MSE(pred, trainy)

            Y_size = trainy.shape

            if iter == 0:
                print("trainy:", trainy.shape)

            optimizer.zero_grad()
            loss_train.backward()
            clip = 5
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            evaluation = evaluate(pred, trainy)
            train_loss.append(loss_train.item())
            train_mape.append(evaluation[0])
            train_rmse.append(evaluation[1])
            train_mae.append(evaluation[2])

            if iter % args.interval == 0:
                log = 'Iter: {:03d}|Train Loss: {:.4f}|Train MAPE: {:.4f}|Train RMSE: {:.4f}|Train MAE: {:.4f}|Time: ' \
                      '{:.4f} '
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1], train_mae[-1], time.time() - t),
                      flush=True)
                t = time.time()
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss, valid_mape, valid_rmse, valid_mae = [], [], [], []
        s1 = time.time()
        for iter, (x_val, y_val) in enumerate(dataloader['val_loader'].get_iterator()):
            # validation data loader iterator init
            inputs_val = torch.Tensor(x_val).to(device)  # x: (64, 24, 207, 2)
            labels_val = torch.Tensor(y_val).to(device)

            if model_type == "GWaveNet":
                inputs_val = inputs_val.transpose(1, 3)
                labels_val = labels_val.transpose(1, 3)
                labels_val = labels_val[:, 0, :, :]
                labels_val = torch.unsqueeze(labels_val, dim=1)

                inputs_val = nn.functional.pad(inputs_val, (1, 0, 0, 0))
                pred_val = model.forward(inputs_val)
                pred_val = pred_val.transpose(1, 3)
                pred_val = scaler.inverse_transform(pred_val)
                loss_valid = loss_gwnet(pred_val, labels_val, 0.0)

            if model_type == "STGCN":
                inputs_val = inputs_val.permute(0, 2, 1, 3)
                labels_val = labels_val[:, :, :, 0].permute(0, 2, 1)
                pred_val = model(A_wave, inputs_val)
                # pred_val = scaler.inverse_transform(pred_val)
                # loss_valid = loss_MSE(pred_val, labels_val)
                loss_valid = loss_stgcn(pred_val, labels_val, 0.0)

            if model_type == "rnn":
                [batch_size, step_size, num_of_vertices, fea_size] = trainx.size()
                inputs_val = inputs_val.permute(0, 2, 1, 3)
                inputs_val = inputs_val.reshape(-1, step_size, fea_size)
                labels_val = labels_val.reshape(-1, 1, fea_size)
                labels_val = labels_val[:, 0, :]
                pred_val = model.loop(inputs_val)
                loss_valid = loss_MSE(pred_val, labels_val)

            # pred_val = scaler.inverse_transform(pred_val)
            optimizer.zero_grad()
            # loss_valid.backward()
            evaluation = evaluate(pred_val, labels_val)

            valid_loss.append(loss_valid.item())
            valid_mape.append(evaluation[0])
            valid_rmse.append(evaluation[1])
            valid_mae.append(evaluation[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(epoch, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mae = np.mean(train_mae)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mae = np.mean(valid_mae)
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
        record_df.round(3).to_csv(f'{args.save}/record.csv')

        log = 'Epoch: {:03d}, Training Time: {:.4f}/epoch,\n' \
              'Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train MAE: {:.4f}, \n' \
              'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid MAE: {:.4f},'
        print(log.format(epoch, (t2 - t1),
                         mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_mae,
                         mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_mae), flush=True)
        print("#" * 20)

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
    if model_type == "GWaveNet":
        target = target.transpose(1, 3)[:, 0, :, :]
    if model_type == "STGCN":
        target = target[:, :, :, 0]
        target = target.transpose(1, 2)

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)  # x: (64, 24, 207, 2)
        testy = torch.Tensor(y).to(device)  # x: (64, 24, 207, 2)

        if model_type == "GWaveNet":
            with torch.no_grad():
                testx = testx.transpose(1, 3)
                pred = model.forward(testx)
                pred = pred.transpose(1, 3)
            outputs.append(pred.squeeze())

        if model_type == "STGCN":
            with torch.no_grad():
                testx = testx.permute(0, 2, 1, 3)
                testy = testy[:, :, :, 0].permute(0, 2, 1)
                pred = model(A_wave, testx)     # (64, 207, 12)
            outputs.append(pred)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:target.size(0), ...]
    amae, amape, armse, test_record = [], [], [], []
    print("=" * 10)
    print("yhat:", yhat.shape)      # yhat: torch.Size([6850, 207, 12])
    print("target:", target.shape)  # target: torch.Size([6850, 207, 12])
    for i in range(Y_size[-1]):
        pred = scaler.inverse_transform(yhat[:, :, i])
        # pred = yhat[:, :, i]
        real_target = target[:, :, i]
        evaluation = evaluate(pred, real_target)
        log = 'Evaluate on test data for horizon {:d}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test MAE: {:.4f}'
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
        print("---  new folder :", path)
    else:
        print("---  There is this folder :", path)


if __name__ == "__main__":
    mkdir(args.save)
    main()
