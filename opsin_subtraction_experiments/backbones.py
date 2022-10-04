import torch
from torch import nn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/np.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, args):

        dim_input = args.dim_input
        dim_output = args.dim_input
        num_inds = args.num_inds
        dim_hidden = args.dim_hidden
        num_heads = args.num_heads
        num_outputs = args.num_traces_per_expt
        ln = args.ln

        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class MultiTraceConv(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        # Initialize layers
        self.feature_encoder = torch.nn.ModuleList([
            DownsamplingBlock(1, 16, 32, 2),
            DownsamplingBlock(16, 16, 32, 1),
            DownsamplingBlock(16, 32, 16, 1),
            DownsamplingBlock(32, 32, 16, 1)
        ])

        self.context_encoder = torch.nn.ModuleList([
            DownsamplingBlock(1, 16, 32, 2),
            DownsamplingBlock(16, 16, 32, 1),
            DownsamplingBlock(16, 32, 16, 1),
            DownsamplingBlock(32, 32, 16, 1)
        ])

        self.ublock1 = UpsamplingBlock(64, 48, 16, 1)
        self.ublock2 = UpsamplingBlock(48 + 32, 32, 16, 1)
        self.ublock3 = UpsamplingBlock(32 + 16, 16, 32, 1)
        self.ublock4 = UpsamplingBlock(16 + 16, 4, 32, 2)
        self.conv = ConvolutionBlock(4, 1, 256, 255, 1, 2)

    def forward(self, x):

        x = torch.squeeze(x)[:, None, :]  # batch x channel x time

        # make feature vector which is aggregate over entire batch
        feats = torch.clone(x)
        for l in self.feature_encoder:
            feats = l(feats)
        dims = feats.shape
        # average over entire batch
        feats = torch.mean(feats, dim=0, keepdim=True)
        # shape of feats now matches shape of inputs
        feats = torch.broadcast_to(feats, dims)

        # Make context embedding, saving outputs to use as skip connections
        context = torch.clone(x)
        skip_inputs = []
        context_sizes = np.zeros(len(self.context_encoder), dtype=int)
        for l in self.context_encoder:
            skip_inputs.append(context)
            context = l(context)

        # Concatenate context and features along channels dimension
        context_plus_features = torch.concat(
            (context, feats), dim=1)  # channels dimension

        # Decoding
        dec1 = self.ublock1(context_plus_features, skip=skip_inputs[3])
        dec2 = self.ublock2(dec1, skip=skip_inputs[2])
        dec3 = self.ublock3(dec2, skip=skip_inputs[1])
        dec4 = self.ublock4(dec3, interp_size=x.shape[-1])

        # Final conv layer
        out = self.conv(dec4)

        return out


class MultiTraceConvAttention(nn.Module):
    ''' Neural waveform demixing U-Net
    '''

    def __init__(self, args=None):
        super().__init__()
        down_filter_sizes = (16, 16, 32, 32)
        up_filter_sizes = (16, 16, 16, 4)

        self.dblock1 = DownsamplingBlock(1, down_filter_sizes[0], 32, 2)
        self.attn1 = MultiChannelAttn(embed_dim=387, num_heads=3)

        self.dblock2 = DownsamplingBlock(
            down_filter_sizes[0], down_filter_sizes[1], 32, 1)
        self.attn2 = MultiChannelAttn(embed_dim=162, num_heads=3)

        self.dblock3 = DownsamplingBlock(
            down_filter_sizes[1], down_filter_sizes[2], 16, 1)
        self.attn3 = MultiChannelAttn(embed_dim=65, num_heads=5)

        self.dblock4 = DownsamplingBlock(
            down_filter_sizes[2], down_filter_sizes[3], 16, 1)
        self.attn4 = MultiChannelAttn(embed_dim=17, num_heads=1)

        self.ublock1 = UpsamplingBlock(
            down_filter_sizes[3], up_filter_sizes[0], 16, 1)
        self.ublock2 = UpsamplingBlock(
            down_filter_sizes[2] + up_filter_sizes[0], up_filter_sizes[1], 16, 1)
        self.ublock3 = UpsamplingBlock(
            down_filter_sizes[1] + up_filter_sizes[1], up_filter_sizes[2], 32, 1)
        self.ublock4 = UpsamplingBlock(
            down_filter_sizes[0] + up_filter_sizes[2], up_filter_sizes[3], 32, 2)

        self.conv = ConvolutionBlock(4, 1, 256, 255, 1, 2)

    def forward(self, x):
        x = torch.squeeze(x)[:, None, :]  # batch x channel x time

        # Encoding
        enc1 = self.dblock1(x)
        enc1 = self.attn1(enc1)

        enc2 = self.dblock2(enc1)
        enc2 = self.attn2(enc2)

        enc3 = self.dblock3(enc2)
        enc3 = self.attn3(enc3)

        enc4 = self.dblock4(enc3)
        enc4 = self.attn4(enc4)

        # Decoding
        dec1 = self.ublock1(enc4, skip=enc3)
        dec2 = self.ublock2(dec1, skip=enc2)
        dec3 = self.ublock3(dec2, skip=enc1)
        dec4 = self.ublock4(dec3, interp_size=x.shape[-1])

        # Final conv layer
        out = self.conv(dec4)

        return out

class MultiChannelAttn(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiChannelAttn, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True, # broadcast over channels as if it's the batch dim
        )

    def forward(self, x):
        x = torch.swapaxes(x, 0, 1)
        x = self.attn(x, x, x, need_weights=False)[0]
        return torch.swapaxes(x, 0, 1)

class DownsamplingBlock(nn.Module):
    ''' DownsamplingBlock
    '''

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DownsamplingBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              dilation=dilation)
        self.decimate = nn.AvgPool1d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.decimate(x))))


class UpsamplingBlock(nn.Module):
    ''' UpsamplingBlock
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, interpolation_mode='linear'):
        super(UpsamplingBlock, self).__init__()

        self.deconv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.interpolation_mode = interpolation_mode

    def forward(self, x, skip=None, interp_size=None):
        if skip is not None:
            up = nn.functional.interpolate(self.relu(self.bn(self.deconv(x))), size=skip.shape[-1],
                                           mode=self.interpolation_mode, align_corners=False)
            return torch.cat([up, skip], dim=1)
        else:
            return nn.functional.interpolate(self.relu(self.bn(self.deconv(x))), size=interp_size,
                                             mode=self.interpolation_mode, align_corners=False)


class ConvolutionBlock(nn.Module):
    ''' ConvolutionBlock
    '''

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation):
        super(ConvolutionBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
