import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from util import *
import argparse


def loss_curve(d):
    tr_val = pd.read_csv(f'{d}', index_col=0)
    # return tr_val[['train_loss', 'valid_loss']]
    return tr_val[['mae']]


def plot_loss_curve(log_dir):
    d = loss_curve(log_dir)
    x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.5)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(-0.5, 11.5)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(2.1, 4.5)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

    plt.plot(d, label='test_mae', c='blue')
    plt.ylabel('Test Loss', fontsize=12)
    plt.xlabel('Predict length(/5min)', fontsize=12)
    plt.grid(True, which='both')
    plt.legend()
    plt.show()


# log_dir = './experiment/improved_Wave/test_metrics.csv'
# plot_loss_curve(log_dir)


def train_loss_curve(d):
    tr_val = pd.read_csv(f'{d}', index_col=0)
    # return tr_val[['train_loss', 'valid_loss']]
    return tr_val[['valid_loss']], tr_val[['train_loss']]


def plot_train_loss_curve(log_dir):
    d0, d1 = train_loss_curve(log_dir)
    plt.plot(d0, label='valid_loss')
    plt.plot(d1, label='train_loss')
    plt.ylabel('Valid Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, which='both')
    plt.legend()
    plt.show()


# train_log = './experiment/improved_Wave/metrics.csv'
# plot_train_loss_curve(train_log)

def plot_baselines_loss():
    # WaveNet_path = './experiment/improved_Wave/metrics.csv'
    WaveNet_path = './experiment/draw_LA/GWaveNet_LA/record.csv'
    STGCN_path = './experiment/draw_LA/STGCN_LA/record.csv'
    STGAT_path = './experiment/draw_LA/STGAT_LA/record.csv'
    WaveNet = pd.read_csv(f'{WaveNet_path}', index_col=0)
    STGCN = pd.read_csv(f'{STGCN_path}', index_col=0)
    STGAT = pd.read_csv(f'{STGAT_path}', index_col=0)

    # valid
    WaveNet_val = WaveNet[['valid_loss']]
    STGCN_val = STGCN[['valid_loss']]
    STGAT_val = STGAT[['valid_loss']]
    plt.tick_params(labelsize=18)
    plt.plot(WaveNet_val, label='WaveNet_val')
    plt.plot(STGCN_val, label='STGCN_val')
    plt.plot(STGAT_val, label='STGAT_val')
    plt.ylabel('Valid Loss', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.grid(True, which='both')
    plt.legend(fontsize=18)
    plt.savefig('LA_baselines_val_loss.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # train
    WaveNet_train = WaveNet[['train_loss']]
    STGCN_train = STGCN[['train_loss']]
    STGAT_train = STGAT[['train_loss']]
    plt.tick_params(labelsize=18)
    plt.plot(WaveNet_train, label='WaveNet_train')
    plt.plot(STGCN_train, label='STGCN_train')
    plt.plot(STGAT_train, label='STGAT_train')
    plt.ylabel('Train Loss', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.grid(True, which='both')
    plt.legend(fontsize=18)
    plt.savefig('LA_baselines_train_loss.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
plot_baselines_loss()


def plot_comparison_gconv():
    no_Gconv_path = './experiment/draw_LA/no_gconv_STGAT_LA/record.csv'
    STGAT_path = './experiment/draw_LA/STGAT_LA/record.csv'
    no_Gconv = pd.read_csv(f'{no_Gconv_path}', index_col=0)
    STGAT = pd.read_csv(f'{STGAT_path}', index_col=0)
    no_Gconv_val = no_Gconv[['valid_loss']]
    STGAT_val = STGAT[['valid_loss']]
    plt.tick_params(labelsize=18)
    plt.plot(no_Gconv_val, label='without_Gconv_val')
    plt.plot(STGAT_val, label='STGAT_val')
    plt.ylabel('Valid Loss', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.grid(True, which='both')
    plt.legend(fontsize=18)
    plt.savefig('LA_without_Gconv.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
# plot_comparison_gconv()



def print_stats(args):
    if '/' not in args.adj_mx:
        full_path = 'data/sensor_graph/' + args.adj_mx
    else:
        full_path = args.adj_mx

    A = load_pickle(full_path)
    adj_mx = A[2]

    degrees = np.sum(adj_mx > 0, 1)
    print(degrees)
    print("adj:", adj_mx.shape)
    print("degrees: ", degrees.shape)
    print(f"Minimum Degree: {np.min(degrees)}")
    print(f"Maximum Degree: {np.max(degrees)}")
    print(f"Average Degree: {np.mean(degrees)}")

    plt.imshow(adj_mx, cmap='hot', interpolation='nearest')
    # plt.title('Adjacency Matrix Heatmap')
    plt.savefig('LA_Matrix_Heatmap.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # print(degrees == 16)
    plt.tick_params(labelsize=18)
    plt.hist(degrees, bins=19, density=0, edgecolor="black", alpha=0.7, color='blue')
    plt.xticks(range(0, max(degrees) + 2, 2))
    # plt.title('LA Degree Distribution')
    plt.xlabel('LA Node Degree', fontsize=24)
    plt.ylabel('Nodes Number', fontsize=24)
    plt.savefig('LA_Degree_Distrib.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adj_mx', type=str, default='adj_mx.pkl',
                        help='Pickle file in data/sensor_graph containing adjacency matrix')
    # parser.add_argument('--adj_mx', type=str, default='adj_mx_bay.pkl',
    #                     help='Pickle file in data/sensor_graph containing adjacency matrix')
    args = parser.parse_args()
    print_stats(args)
