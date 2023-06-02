from matplotlib.cbook import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import os
from torch.nn.functional import dropout
from models.modeling import utils
from models.modeling.backbone_vit.vit_model import vit_base_patch16_224 as create_model

from os.path import join
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init
Norm =nn.LayerNorm


def trunc_normal_(tensor, mean=0, std=.01):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class SimpleReasoning(nn.Module):
    def __init__(self, np,ng):
        super(SimpleReasoning, self).__init__()
        self.hidden_dim= np//ng 
        self.fc1 = nn.Linear(np, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, np)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.act=nn.GELU()

    def forward(self, x):
        x_1 = self.fc1(self.avgpool(x).flatten(1)) 
        x_1 = self.act(x_1)
        x_1 = F.sigmoid(self.fc2(x_1)).unsqueeze(-1)
        x_1 = x_1*x + x
        return x_1

class Tokenmix(nn.Module):
    def __init__(self, np):
        super(Tokenmix, self).__init__()
        dim =196
        hidden_dim = 512
        dropout = 0.
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.act=nn.GELU()
        self.norm = nn.LayerNorm(np)
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout))
    def forward(self, x):
        redisual =  x
        x=self.norm(x)
        x = rearrange(x, "b p c -> b c p")
        x = self.net(x)
        x = rearrange(x, "b c p-> b p c")
        out = redisual+x
        return out


class AnyAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super(AnyAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = Norm(dim), Norm(dim), Norm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = dim ** (-0.5)
        self.act=nn.ReLU()
        self.proj = nn.Linear(dim, dim)
    def get_qkv(self, q, k, v):
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v
    def forward(self, q=None, k=None, v=None):
        q, k, v = self.get_qkv(q, k, v)
        attn = torch.einsum("b q c, b k c -> b q k", q, k)
        attn = self.act(attn)
        attn *= self.scale
        attn_mask = F.softmax(attn, dim=-1)
        out = torch.einsum("b q k, b k c -> b q c", attn_mask, v.float())
        out = self.proj(out)
        return attn, out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.fc1.bias, -bound1, bound1)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.fc2.bias, -bound2, bound2)



class Block(nn.Module):
    def __init__(self, dim, ffn_exp=4, drop_path=0.1, num_heads=1, num_parts=0,num_g=6):
        super(Block, self).__init__()
        self.dec_attn = AnyAttention(dim, True)
        self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=nn.GELU, norm_layer=Norm)
        self.drop_path = nn.Identity()
        self.reason = Tokenmix(dim)
        self.enc_attn = AnyAttention(dim, True)
        self.group_compact = SimpleReasoning(num_parts,num_g)
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)
        self.enc_ffn = Mlp(dim, hidden_features=dim, act_layer=nn.GELU) 


    def forward(self, x, parts=None):
        x = rearrange(x, "b c p -> b p c")
        attn_0,attn_out = self.enc_attn(q=parts, k=x, v=x)
        attn_0=self.maxpool1d(attn_0).flatten(1)
        parts1 = parts + attn_out
        parts2 = self.group_compact(parts1)
        if self.enc_ffn is not None:
            parts_out = parts2 + self.enc_ffn(parts2)+parts1
        parts_d= parts+parts_out
        attn_1,attn_out = self.enc_attn(q=parts_d, k=x, v=x)
        attn_1=self.maxpool1d(attn_1).flatten(1)
        parts1_d = parts_d + attn_out
        parts_comp = self.group_compact(parts1_d)
        if self.enc_ffn is not None:
            parts_in = parts_comp + self.enc_ffn(parts_comp)+parts1_d
        attn_mask,feats = self.dec_attn(q=x, k=parts_in, v=parts_in)
        feats = x + feats
        feats = self.reason(feats)
        feats = feats + self.ffn1(feats)
        feats = rearrange(feats, "b p c -> b c p")
        return feats,attn_0,attn_1

class PSVMANet(nn.Module):
    def __init__(self, basenet, c,
                 attritube_num, cls_num, ucls_num, group_num, w2v,
                 scale=20.0, device=None):

        super(PSVMANet, self).__init__()
        self.attritube_num = attritube_num
        self.group_num=group_num
        self.feat_channel = c
        self.batch =10
        self.cls_num= cls_num
        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.w2v_att = torch.from_numpy(w2v).float().to(device)
        self.W = nn.Parameter(trunc_normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),
                              requires_grad=True)
        self.V = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, self.attritube_num)),
                              requires_grad=True)
        assert self.w2v_att.shape[0] == self.attritube_num
        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

        self.backbone_patch = nn.Sequential(*list(basenet.children()))[0]
        self.backbone_drop= nn.Sequential(*list(basenet.children()))[1]
        self.backbone_0 = nn.Sequential(*list(basenet.children()))[2][:-1]
        self.backbone_1 = nn.Sequential(*list(basenet.children()))[2][-1]

        self.drop_path = 0.1

        self.cls_token = basenet.cls_token
        self.pos_embed = basenet.pos_embed

        self.cat = nn.Linear(self.attritube_num*self.feat_channel, attritube_num)
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.CLS_loss = nn.CrossEntropyLoss()
        self.Reg_loss = nn.MSELoss()

        self.blocks = Block(self.feat_channel,
                  num_heads=1,
                  num_parts=self.attritube_num,
                  num_g=self.group_num,
                  ffn_exp=4,
                  drop_path=0.1)

    def compute_score(self, gs_feat,seen_att,att_all):
        gs_feat = gs_feat.view(self.batch, -1)
        gs_feat_norm = torch.norm(gs_feat, p=2, dim = 1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)
        temp_norm = torch.norm(att_all, p=2, dim=1).unsqueeze(1).expand_as(att_all)
        seen_att_normalized = att_all.div(temp_norm + 1e-5)
        score_o = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)  
        d, _ = seen_att.shape
        score_o = score_o*self.scale
        if d == self.cls_num:
            score = score_o
        if d == self.scls_num:
            score = score_o[:, :d]
            uu = self.ucls_num
            if self.training:
                mean1 = score_o[:, :d].mean(1)
                std1 = score_o[:, :d].std(1)
                mean2 = score_o[:, -uu:].mean(1)
                std2 = score_o[:, -uu:].std(1)
                mean_score = F.relu(mean1 - mean2)
                std_score = F.relu(std1 - std2)
                mean_loss = mean_score.mean(0) + std_score.mean(0)
                return score, mean_loss
        if d == self.ucls_num:
            score = score_o[:, -d:]
        return score,_
    
    def forward(self, x, att=None, label=None, seen_att=None, att_all=None):
        self.batch = x.shape[0]
        parts = torch.einsum('lw,wv->lv', self.w2v_att, self.W)
        parts = parts.expand(self.batch, -1, -1)
        patches = self.backbone_patch(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        patches = torch.cat((cls_token, patches), dim=1) 
        feats_0 = self.backbone_drop(patches + self.pos_embed)
        feats_0 = self.backbone_0(feats_0)
        feats_in = feats_0[:, 1:, :]
        feats_out, att_0, att_1 = self.blocks(feats_in.transpose(1,2),parts=parts)
        patches_1 = torch.cat((cls_token, feats_out.transpose(1,2)), dim=1) 
        feats_1 = self.backbone_1(patches_1+self.pos_embed)
        feats_1 = feats_1[:, 1:, :]
        feats_1, att_2, att_3 = self.blocks(feats_1.transpose(1, 2), parts=parts)

        feats=feats_1
        out = self.avgpool1d(feats.view(self.batch, self.feat_channel, -1)).view(self.batch, -1)
        out = torch.einsum('bc,cd->bd', out, self.V)
        score, b = self.compute_score(out, seen_att, att_all)

        if not self.training:
            return score

        Lreg1 = self.Reg_loss(att_0, att)+self.Reg_loss(att_1, att)+self.Reg_loss(att_2, att)+self.Reg_loss(att_3, att)
        Lcls = self.CLS_loss(score, label)
        scale = self.scale.item()
        loss_dict = {
            'Reg_loss': Lreg1,
            'Cls_loss': Lcls,
            'scale': scale,
            'bias_loss': b
        }

        return loss_dict


def build_PSVMANet(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    group_num = info["g"] 
    c,w,h = 768, 14, 14
    scale = cfg.MODEL.SCALE
    vit_model = create_model(num_classes=-1)
    vit_model_path = "../../pretrain_model_vit/vit_base_patch16_224.pth"
    weights_dict = torch.load(vit_model_path)
    del_keys = ['head.weight', 'head.bias'] if vit_model.has_logits \
        else ['head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    vit_model.load_state_dict(weights_dict, strict=False)
    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return PSVMANet(basenet=vit_model,
                  c=c,scale=scale,
                  attritube_num=attritube_num,
                  group_num=group_num, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)