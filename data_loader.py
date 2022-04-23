import json
import torch
import copy
import numpy as np
from utils import get_config

class LawDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, dataset, batch_size, facts_dict, d_type='validation'):
        self.dataset = dataset + "/"
        self.batch_size = batch_size # 16
        self.facts_dict = facts_dict
        self.ptr = 0
        self.data = []

        if d_type == 'train':
            data_file = self.dataset + 'train_set.json'
        elif d_type == 'validation':
            data_file = self.dataset + 'val_slice.json'
        else:
            data_file = self.dataset + 'test_slice.json'
        
        config_file = self.dataset + 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        lawyer_n, _, field_n = get_config(config_file, "data")
        self.field_dim = int(field_n)
        self.lawy_dim = int(lawyer_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None, None, None, None, None, None, None
        input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode, ys = [], [], [], [], [], [], [], [], [], []
        member_max_len = 0
        cutoff_len = 510
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            if len(log['member']) > member_max_len:
                member_max_len = len(log['member'])
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            field_emb = [0.] * self.field_dim
            for field_code in log['knowledge_code']:
                field_emb[field_code - 1] = 1.0
            y = log['score']
            input_lawy_ids.append(log['user_id'] - 1)
            input_case_ids.append(log['exer_id'] - 1)
            input_field_embs.append(field_emb)
            pla_num = 0
            def_num = 0
            pla_emb = []
            def_emb = []
            if log['tag'] == 0:
                for lawyer in log['member']:
                    if lawyer > 0:
                        pla_num += 1
                        pla_emb.append(lawyer)
                    else:
                        def_num += 1
                        def_emb.append(lawyer * (-1))
            else:
                for lawyer in log['member']:
                    if lawyer > 0:
                        def_num += 1
                        def_emb.append(lawyer)
                    else:
                        pla_num += 1
                        pla_emb.append(lawyer * (-1))
            for i in range(len(pla_emb), member_max_len):
                pla_emb.append(0)
            for i in range(len(def_emb), member_max_len):
                def_emb.append(0)
            
            
            # fact_encode = self.facts_dict[self.idx2case[log['exer_id']]]
            fact_encode = self.facts_dict[str(log['exer_id'])]
            input_pla_nums.append(pla_num)
            input_def_nums.append(def_num)
            input_pla_embs.append(pla_emb)
            input_def_embs.append(def_emb)
            input_factions.append(log['tag'])
            facts_encode.append(fact_encode)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_lawy_ids), torch.LongTensor(input_case_ids), torch.Tensor(input_field_embs), torch.Tensor(input_factions), torch.LongTensor(input_pla_nums), torch.LongTensor(input_def_nums), torch.LongTensor(input_pla_embs), torch.LongTensor(input_def_embs), torch.Tensor(facts_encode), torch.Tensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, dataset, facts_dict, d_type='validation'):
        self.dataset = dataset + "/"
        self.d_type = d_type
        self.ptr = 0
        self.data = []

        if d_type == 'validation':
            data_file = self.dataset + 'val_set.json'
        else:
            data_file = self.dataset + 'test_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(self.dataset + config_file) as i_f:
            i_f.readline()
            lawyer_n, _, field_n = i_f.readline().split(',')
            self.field_dim = int(field_n)
            self.lawy_dim = int(lawyer_n) + 1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None, None, None, None, None, None, None, None, None
        if self.data[self.ptr]['logs'] == []:
            self.ptr += 1
            return None, None, None, None, None, None, None, None, None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, token_ids, attn_masks, token_type_ids, ys = [], [], [], [], [], [], [], [], [], [], [], []
        member_max_len = 0
        fact_max_len = 0
        cutoff_len = 500
        for log in logs:
            # fact = self.facts_dict[self.idx2case[log['exer_id']]]
            fact = self.facts_dict[str(log['exer_id'])]
            if len(log['member']) > member_max_len:
                member_max_len = len(log['member'])
            if len(fact) > fact_max_len:
                if len(fact) > cutoff_len:
                    fact_max_len = cutoff_len
                else:
                    fact_max_len = len(fact)
        for log in logs:
            field_emb = [0.] * self.field_dim
            for field_code in log['knowledge_code']:
                field_emb[field_code - 1] = 1.0
            y = log['score']
            input_lawy_ids.append(user_id - 1)
            input_case_ids.append(log['exer_id'] - 1)
            input_field_embs.append(field_emb)
            pla_num = 0
            def_num = 0
            pla_emb = []
            def_emb = []
            if log['tag'] == 0:
                for lawyer in log['member']:
                    if lawyer > 0:
                        pla_num += 1
                        pla_emb.append(lawyer)
                    else:
                        def_num += 1
                        def_emb.append(lawyer * (-1))
            else:
                for lawyer in log['member']:
                    if lawyer > 0:
                        def_num += 1
                        def_emb.append(lawyer)
                    else:
                        pla_num += 1
                        pla_emb.append(lawyer * (-1))
            for i in range(len(pla_emb), member_max_len):
                pla_emb.append(0)
            for i in range(len(def_emb), member_max_len):
                def_emb.append(0)
            # fact = self.facts_dict[self.idx2case[log['exer_id']]]
            fact = self.facts_dict[str(log['exer_id'])]
            encoded_pair = self.tokenizer(fact, padding='max_length', truncation=True, max_length=cutoff_len, return_tensors='pt')
            token_id = encoded_pair['input_ids'].squeeze(0).numpy().tolist()
            attn_mask = encoded_pair['attention_mask'].squeeze(0).numpy().tolist()
            token_type_id = encoded_pair['token_type_ids'].squeeze(0).numpy().tolist()
#             if pla_num == 0:
#                 pla_num += 1
#             if def_num == 0:
#                 def_num += 1
            input_pla_nums.append(pla_num)
            input_def_nums.append(def_num)
            input_pla_embs.append(pla_emb)
            input_def_embs.append(def_emb)
            input_factions.append(log['tag'])
            token_ids.append(token_id)
            attn_masks.append(attn_mask)
            token_type_ids.append(token_type_id)
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_lawy_ids), torch.LongTensor(input_case_ids), torch.Tensor(input_field_embs), torch.Tensor(input_factions), torch.LongTensor(input_pla_nums), torch.LongTensor(input_def_nums), torch.LongTensor(input_pla_embs), torch.LongTensor(input_def_embs), torch.LongTensor(token_ids), torch.LongTensor(attn_masks), torch.LongTensor(token_type_ids), torch.Tensor(ys)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
