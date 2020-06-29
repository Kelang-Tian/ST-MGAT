import torch.nn.functional as F
import math
from dgl.nn.pytorch import edge_softmax, GATConv
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
import dgl

class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        # print("X.shape:",X.shape)
        temp = self.conv1(X)
        temp += torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class stgat(nn.Module):
    def __init__(self, g, run_gconv, dropout=0.3, in_dim=2, out_dim=12,
                 residual_channels=40, dilation_channels=40,
                 skip_channels=320, end_channels=640,
                 kernel_size=2, blocks=4, layers=2):
        super().__init__()
        print("========batch_g_gat_2l===========")
        self.g = g
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.run_gconv = run_gconv

        # d = (d - kennel_size + 2 * padding) / stride + 1
        self.start_conv = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.cat_feature_conv = nn.Conv2d(in_channels=1,
                                          out_channels=residual_channels,
                                          kernel_size=(1, 1))

        depth = list(range(blocks * layers))

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = ModuleList([Conv1d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])

        self.gat_layers = nn.ModuleList()
        self.gat_layers1 = nn.ModuleList()
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()

        heads = 8
        feat_drop = 0.6
        attn_drop = 0.6
        negative_slope = 0.2
        receptive_field = 1
        # time_len change: 12/10/9/7/6/4/3/1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1  # dilation
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                # batch, channel, height, width
                # N,C,H,W
                # d = (d - kennel_size + 2 * padding) / stride + 1
                # H_out = [H_in + 2*padding[0] - dilation[0]*(kernal_size[0]-1)-1]/stride[0] + 1
                # W_out = [W_in + 2*padding[1] - dilation[1]*(kernal_size[1]-1)-1]/stride[1] + 1

                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gat_layers.append(GATConv(
                    dilation_channels*(14 - receptive_field),
                    dilation_channels*(14 - receptive_field),
                    heads, feat_drop, attn_drop, negative_slope,
                    residual=False, activation=F.elu))
                self.gat_layers1.append(GATConv(
                    dilation_channels * (14 - receptive_field),
                    dilation_channels * (14 - receptive_field),
                    heads, feat_drop, attn_drop, negative_slope,
                    residual=False, activation=F.elu))

        self.receptive_field = receptive_field
        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)

    def forward(self, x):
        # Input shape is (bs, features, n_nodes, n_timesteps)
        print("===131 x.shape: ", x.shape)  # torch.Size([64, 2, 207, 13])

        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
            # print("===136x.shape:", x.shape)

        x1 = self.start_conv(x[:, [0]])
        x2 = F.leaky_relu(self.cat_feature_conv(x[:, [1]]))
        # print("x1", x1.shape)       # [64, 40, 207, 13]
        # print("x2", x2.shape)       # [64, 40, 207, 13]
        # batch, channel, height, width
        x = x1 + x2
        # print("x.shape:", x.shape)  # torch.Size([64, 40, 207, 13])
        skip = 0

        # STGAT layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            # print("{}=== filter:{}, gate:{}===".format(i, filter.shape, gate.shape))
            # print("175residual:", residual.shape)
            # filter:[64, 40, 207, 12/10/9/7/6/4/3/1]

            # parametrized skip connection
            s = self.skip_convs[i](x)  # [64, 320, 207, 12/10/9]

            try:  # if i > 0 this works
                skip = skip[:, :, :, -s.size(3):]  # TODO(SS): Mean/Max Pool?
                # print("==181 skip[:, :, :,  -s.size(3):]: ", skip.shape)
                # ([64, 320, 207, /10/9]
            except:
                skip = 0
            skip = s + skip
            # print("183skip:", skip.shape)  # [64, 320, 207, 12/10/9]
            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            # graph conv and mix
            if self.run_gconv:
                [batch_size, fea_size, num_of_vertices, step_size] = x.size()
                # print("157 x:", x.shape)
                batched_g = dgl.batch(batch_size * [self.g])
                h = x.permute(0, 2, 1, 3).reshape(batch_size*num_of_vertices, fea_size*step_size)
                # print("159 h:", h.shape)
                h = self.gat_layers[i](batched_g, h).mean(1)
                h = self.gat_layers1[i](batched_g, h).mean(1)

                # print("164 h:", h.shape)
                gc = h.reshape(batch_size, num_of_vertices, fea_size, -1)
                # print("166 gc:", gc.shape)
                graph_out = gc.permute(0, 2, 1, 3)
                x = x + graph_out
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]  # TODO(SS): Mean/Max Pool?
            # print("267 x_:", x.shape)  # [64, 40, 207, 12]
            x = self.bn[i](x)
            # print("x_last:", x.shape)  # [64, 40, 207, 12/10/9]

        x = F.relu(skip)  # ignore last X
        # print("201 skip: ", skip.shape)
        x = F.relu(self.end_conv_1(x))
        # print("203 x:", x.shape)
        x = self.end_conv_2(x)  # downsample to (bs, seq_length, 207, nfeatures)
        # print("return x:", x.shape)
        return x
