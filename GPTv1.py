#.ipynb 를 통해 dataset과 vocab을 만든다.
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import pandas as pd
from IPython.display import display
from tqdm import tqdm, trange
import sentencepiece as spm
import wget

import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_file = "./web-crawler/kowiki/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

print(f'vocab : {vocab}')

""" configuration json을 읽어들이는 class"""
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read(0))
            return Config(config)

config = Config({
    "n_dec_vocab" : len(vocab),
    "n_dec_seq" : 256,
    "n_layer" : 6,
    "d_hidn" : 256,
    "i_pad" : 0,
    "d_ff" : 1024,
    "n_head" : 4,
    "d_head" : 64,
    "dropout" : 0.1,
    "layer_norm_epsilon" : 1e-12,
    "device" : 'cuda'
})

"""attention pad mask"""
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    #(bs, q_seq_len), (bs, k_seq_len)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    #(bs, k_seq_len) -> (bs, 1, k_seq_len) -> (bs, q_seq_len, k_seq_len) : 
    #(1, k_seq_len)를 아래로, q_seq_len 만큼 복사
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask

"""attention decoder mask"""
def get_attn_decoder_mask(seq):
    #(bs, seq_len, seq_len)
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask

"""scale dot product attention"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5) #d_head : head dimension, n_head * d_head = d_hidn

    def forward(self, Q, K, V, attn_mask):
        #(bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1,-2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        #(bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        #(bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)
        #(bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_k_seq)
        return context, attn_prob
    
"""multi-head attention"""
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, K, Q, V, attn_mask):
        batch_size = Q.size(0)
        #(bs, n_head, n_q_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

        #(bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.config.n_head,1 ,1)
        
        #(bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        #(bs, n_q_seq, n_head * d_head)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head)
        #(bs, n_head, n_q_seq, d_hidn)
        output = self.linear(context)
        output = self.dropout(output)
        #(bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob
    
"""feed forward"""
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels = self.config.d_hidn, out_channels = self.config.d_ff, kernel_size = 1)
        self.conv2 = nn.Conv1d(in_channels = self.config.d_ff, out_channels = self.config.d_hidn, kernel_size = 1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        #(bs, d_ff, n_seq)
        output = self.active(self.conv1(inputs.transpose(1,2)))
        #(bs, n_seq, d_hidn)
        output = self.conv2(output).transpose(1,2)
        output = self.dropout(output)
        #(bs, n_seq, d_hidn)
        return output
        
"""decoder layer"""
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps = self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps = self.config.layer_norm_epsilon)

    def forward(self, dec_inputs, self_attn_mask):
        #(bs, n_dec_seq, d_hidn), (bs, n_head, n_dec,seq, n_dec_seq)
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        #(bs, n_dec_seq, d_hidn)
        ffn_outputs = self.pos_ffn(self_att_outputs)
        ffn_outputs = self.layer_norm3(self_att_outputs + ffn_outputs)
        #(bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq), (bs, n_head, n_dec_seq, n_enc_seq)
        return ffn_outputs, self_attn_prob
    
"""decoder"""
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)
        self.pos_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, dec_inputs):
        positions = torch.arange(dec_inputs.size(1), device = dec_inputs.device, dtype = dec_inputs.dtype).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
        pos_mask = dec_inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)

        #(bs, n_dec_seq, d_hidn)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions)

        #(bs, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        #(bs, n_dec_seq, n_dec_seq)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
        #(bs, n_dec_seq, n_dec_seq)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask),0)

        self_attn_probs = []
        for layer in self.layers:
            #(bs, n_dec_seq, d_hidn), (bs, n_dec_seq, n_dec_seq)
            dec_outputs, self_attn_prob = layer(dec_outputs, dec_self_attn_mask)
            self_attn_probs.append(self_attn_prob)

        #(bs, n_dec_seq, d_hidn), [(bs, n_dec_seq, n_dec_seq)])
        return dec_outputs, self_attn_probs
    
"""gpt"""
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = Decoder(self.config)

    def forward(self, dec_inputs):
        #(bs, n_seq, d_hidn), [(bs, n_head, n_dec_seq, n_dec_seq)]
        dec_outputs, dec_self_attn_probs = self.decoder(dec_inputs)
        #(bs, n_dec_seq, n_dec_vocab), [(bs, n_head, n_dec_seq, n_dec_seq)]
        return dec_outputs, dec_self_attn_probs

    def save(self, epoch, loss ,path):
        torch.save({
            "epoch" : epoch,
            "loss" : loss,
            "state_dict" : self.state_dict()
        },path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]
    
"""gpt pre-train"""
class GPTPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.gpt = GPT(self.config)
        #lm
        self.projection_lm = nn.Linear(self.config.d_hidn, self.config.n_dec_vocab, bias = False)
        self.projection_lm.weight = self.gpt.decoder.dec_emb.weight

    def forward(self, dec_inputs):
        #(bs, n_dec_seq, d_hidn), [(bs, n_head, n_dec_seq, n_dec_seq)]
        dec_outputs, dec_self_attn_probs = self.gpt(dec_inputs)
        #(bs, n_dec_seq, n_dec_vocab)
        logits_lm = self.projection_lm(dec_outputs)
        # (bs, n_dec_seq - 1, n_dec_vocab), (bs, n_output), [(bs, n_head, n_dec_seq, n_dec_seq)]
        # logits_lm[:, :-1, :] -> input을 [<sos> ~ <eos>] 로 넣기 때문에 output은 [~ <eos>, trash]로 나온다. 해서 마지막은 버리고 output으로 가져옴
        return logits_lm[:, :-1, :].contiguous(), dec_self_attn_probs
    
""" pretrain 데이터셋 """
class PretrainDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.sentences = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            with tqdm(total=line_cnt, desc=f"Loading") as pbar:
                for i, line in enumerate(f):
                    instance = json.loads(line)
                    self.sentences.append([vocab.piece_to_id(p) for p in instance["tokens"]])
                    pbar.update(1)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        return (torch.tensor(self.sentences[item]), torch.tensor(item))
    
""" pretrain data collate_fn """
def pretrin_collate_fn(inputs):
    dec_inputs, item = list(zip(*inputs))

    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

    batch = [
        dec_inputs,
        torch.stack(item, dim=0),
    ]
    return batch

""" pretrain 데이터 로더 """
batch_size = 64#128로 하면 24G도 튕긴다.
dataset = PretrainDataSet(vocab, f"./web-crawler/kowiki/kowiki_gpt.json")
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn= pretrin_collate_fn)

""" 모델 epoch 학습 """
def train_epoch(config, epoch, model, criterion_lm, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, value in enumerate(train_loader):
            dec_inputs, _ = map(lambda v: v.to(config.device), value)
            #<sos> 제외, target은 [~ <eos>]
            labels_lm = dec_inputs[:, 1:].contiguous()

            optimizer.zero_grad()
            #output은 [~ <eos>]
            outputs = model(dec_inputs)
            logits_lm = outputs[0]

            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
            loss = loss_lm 

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(config)

learning_rate = 5e-5
n_epoch = 10

import gc
gc.collect()
torch.cuda.empty_cache()

model = GPTPretrain(config)

save_pretrain = f"./web-crawler/kowiki/save_gpt_pretrain.pth"
best_epoch, best_loss = 0, 0
# if os.path.isfile(save_pretrain): # pretrain된 weight가 있으면 불러올 수 있다. 
#     best_epoch, best_loss = model.gpt.load(save_pretrain)
#     print(f"load pretrain from: {save_pretrain}, epoch={best_epoch}, loss={best_loss}")
#     best_epoch += 1

model.to(config.device)

criterion_lm = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
offset = best_epoch
for step in trange(n_epoch, desc="Epoch"):
    epoch = step + offset
    loss = train_epoch(config, epoch, model, criterion_lm, optimizer, train_loader)
    losses.append(loss)
    model.gpt.save(epoch, loss, save_pretrain)



with open('./results_pre_training.txt','w') as f:
    f.write("loss\n")
    for i in range(len(losses)):
        f.write(str(losses[i]))
        f.write('\n')