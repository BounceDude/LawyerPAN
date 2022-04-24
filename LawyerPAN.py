import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
import json


class LawyerPAN(nn.Module):
    def __init__(self, lawyer_n, case_n, field_n, batch_size, device, hidden_size=768):
        super(LawyerPAN, self).__init__()

        self.field_dim = field_n
        self.case_n = case_n
        self.emb_num = lawyer_n
        self.lawy_dim = self.field_dim
        self.prednet_input_len = self.field_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.device = device
        self.batch_size = batch_size
        self.faction = nn.Parameter(torch.randn(1, self.field_dim))
        self.LC1 = nn.Linear(hidden_size, self.field_dim)
        self.LC2 = nn.Linear(hidden_size, self.field_dim)
        self.disc_LC = nn.Linear(self.field_dim, 1)

        # network structure
        self.text_attn = Attention(hidden_size, self.lawy_dim)
        self.lawyer_emb = nn.Embedding(self.emb_num, self.lawy_dim)
        self.e_discrimination = nn.Embedding(self.case_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len * 2, self.prednet_len1, bias=False)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2, bias=False)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1, bias=False)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name and 'bert' not in name:
                nn.init.xavier_normal_(param)

    def forward(self, lawy_id, case_id, field_id, factions, pla_num, def_num, pla_emb, def_emb, facts_encode):
        '''
        :param lawy_id: LongTensor
        :param exer_id: LongTensor
        :param field_id: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        self.batch_size = len(lawy_id)
        lawy_emb = torch.sigmoid(self.lawyer_emb(lawy_id))
        plaintiff_emb = torch.sigmoid(self.team_concat(pla_emb))
        defendant_emb = torch.sigmoid(self.team_concat(def_emb))
        plaintiff_diff = torch.sigmoid(self.LC1(facts_encode))
        defendant_diff = torch.sigmoid(self.LC2(facts_encode))
        plaintiff_discr = torch.sigmoid(self.disc_LC(self.team_maxpool(pla_emb) - self.team_minpool(pla_emb)))
        defendant_discr = torch.sigmoid(self.disc_LC(self.team_maxpool(def_emb) - self.team_minpool(def_emb)))

        # prednet
        factions = factions.unsqueeze(1)
        input_x_i = (plaintiff_discr * (1 - factions) + defendant_discr * factions) * (lawy_emb - plaintiff_diff * (1 - factions) - defendant_diff * factions) * field_id
        adversary = plaintiff_emb - plaintiff_diff - defendant_emb + defendant_diff
        input_x_g = (plaintiff_discr * (1 - factions) + defendant_discr * factions) * (adversary * (1 - factions) - adversary * factions) * field_id
        input_x = torch.cat([input_x_i, input_x_g * torch.pow(-1, factions)], 1)
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def team_concat(self, emb):
        b = self.lawyer_emb(emb)
        mask = emb.unsqueeze(2).expand(-1, -1, self.lawy_dim).float()
        c = b.masked_fill(mask == 0, -1e9)
        d = torch.softmax(c, dim=1)
        e = torch.sum(b * d, dim=1)
        return e

    def team_maxpool(self, emb):
        pooling = nn.MaxPool2d((emb.size()[1], 1), stride=1)
        b = self.lawyer_emb(emb)
        mask = emb.unsqueeze(2).expand(-1, -1, self.lawy_dim).float()
        c = b.masked_fill(mask == 0, -1e9)
        d = pooling(c).squeeze(1)
        return d

    def team_minpool(self, emb):
        b = self.lawyer_emb(emb)
        mask = emb.unsqueeze(2).expand(-1, -1, self.lawy_dim).float()
        c = b.masked_fill(mask == 0, 10000)
        d = torch.min(c, dim=1, keepdim=False)[0]
        return d

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)
        self.text_attn.fc.apply(clipper)

    def get_lawyer_proficiency(self, stu_id):
        prof_emb = torch.sigmoid(self.lawyer_emb(lawy_id))
        return prof_emb.data


class Attention(nn.Module):
    def __init__(self, dim1, dim2):
        super(Attention, self).__init__()
        self.fc = nn.Linear(dim1, dim2)

    def forward(self, query, context):
        query = self.fc(query)
        attention = torch.bmm(context, query.transpose(1, 2))
        attention_sm = torch.softmax(attention, dim=-1)
        context_vec = torch.bmm(attention_sm, query)
        return context_vec, attention, attention_sm


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
