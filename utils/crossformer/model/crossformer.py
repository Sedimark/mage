import sys
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange, repeat

from default_repo.utils.crossformer.model.layers.attention import TwoStageAttentionLayer
from default_repo.utils.crossformer.model.layers.encoder import Encoder
from default_repo.utils.crossformer.model.layers.decoder import Decoder
from default_repo.utils.crossformer.model.layers.embedding import PositionEmbedding, ValueEmebedding, DSWEmbedding

from default_repo.utils.crossformer.metrics import metric

from math import ceil

class TimeSeriesTransformer(nn.Module): 
    def __init__(self, data_dim, in_len, out_len, seg_len, window_size = 4,
                factor=10, model_dim=512, feedforward_dim = 1024, heads_num=8, blocks_num=3, 
                dropout=0.0, baseline = False, learning_rate=1e-4, batch=32):
        super(TimeSeriesTransformer, self).__init__()

        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = window_size

        self.baseline = baseline

        
        # Segment Number alculation
        self.in_seg_num = ceil(1.0 * in_len / seg_len)
        self.out_seg_num = ceil(1.0 * out_len / seg_len)

        # Encode Embedding & Encoder
        # self.enc_embedding = DSWEmbedding(seg_len=self.seg_len, model_dim=model_dim, data_dim=data_dim, seg_num=self.in_seg_num)
        self.enc_embedding = ValueEmebedding(seg_len=self.seg_len, model_dim=model_dim)
        self.enc_pos = nn.Parameter(torch.randn(1, data_dim, (self.in_seg_num), model_dim))
        self.norm = nn.LayerNorm(model_dim)
        self.encoder = Encoder(blocks_num=blocks_num, model_dim=model_dim, window_size=window_size, depth=1, seg_num=self.in_seg_num, factor=factor, heads_num=heads_num, feedforward_dim=feedforward_dim, dropout=dropout)

        # Decode Embedding & Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.out_seg_num), model_dim))
        self.decoder = Decoder(seg_len=self.seg_len, model_dim=model_dim, heads_num=heads_num, depth=1, feedforward_dim=feedforward_dim, dropout=dropout, out_segment_num=self.out_seg_num, factor=factor)

    def forward(self, x_seq):
        if (self.baseline):
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_seg_num*self.seg_len != self.in_len):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, (self.seg_len*self.in_seg_num-self.in_len), -1), x_seq), dim=1)
        
        # x_seq = self.enc_embedding(x_seq)
        x_seq = self.enc_embedding(x_seq)
        x_seq += self.enc_pos
        x_seq = self.norm(x_seq)

        enc_out = self.encoder(x_seq)
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)


        return base + predict_y[:, :self.out_len, :]