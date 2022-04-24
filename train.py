import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json
import sys
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from data_loader import LawDataLoader
from LawyerPAN import LawyerPAN
from utils import get_config
import time
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='LawyerPAN')
parser.add_argument('--gpu',
                    default='cuda:0',
                    type=str,
                    help='gpu')
parser.add_argument('--epochs',
                    default=10,
                    type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch',
                    default=64,
                    type=int,
                    help='batch size')
parser.add_argument('--lr', '--learning-rate',
                    default=0.0001,
                    type=float,
                    help='initial learning rate')
args = parser.parse_args()

# can be changed according to command parameter
device = torch.device((args.gpu) if torch.cuda.is_available() else 'cpu')
epoch_n = args.epochs
batch_size = args.batch
lr = args.lr

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print('Args:')
for k in list(vars(args).keys()):
    print("\t{}: {}".format(k, vars(args)[k]))

path = "./data/"
# can be changed according to config.txt
lawyer_n, case_n, field_n = get_config(path + "config.txt", "data")

model_path = "./model/"
if not os.path.exists(model_path):
    os.makedirs(model_path)

print("Facts loading...")
with open(path + "fact2vec.json", encoding='utf8') as i_f:
    facts_dict = json.load(i_f)
print("Facts loading completed!")
performances = []

def train():
    data_loader = LawDataLoader(path, batch_size, facts_dict, 'train')
    model = LawyerPAN(lawyer_n, case_n, field_n, batch_size, device)
    model = model.to(device)
    validate(model, 0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('training model...')

    loss_function = nn.BCELoss()
    global_step = 0
    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            global_step += 1
            batch_count += 1
            input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode, labels = data_loader.next_batch()
            if labels is None:
                continue
            input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode, labels = input_lawy_ids.to(device), input_case_ids.to(device), input_field_embs.to(device), input_factions.to(device), input_pla_nums.to(device), input_def_nums.to(device), input_pla_embs.to(device), input_def_embs.to(device), facts_encode.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode)
            output = output.view(-1)

            loss = loss_function(output+1e-10, labels)
            loss.backward()
            optimizer.step()
            model.apply_clipper()

            ctime = int(time.time())
            ctime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ctime))
            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f, time: %s' % (epoch + 1, batch_count + 1, running_loss / batch_count, ctime_str))
                running_loss = 0.0

        # validate and save current model every epoch
        save_snapshot(model, model_path + 'model_epoch' + str(epoch + 1))
        validate(model, epoch + 1)

    # test
    top_performance = sorted(performances, key=lambda x: x[1], reverse=True)[0]
    test(top_performance[0])

def validate(saved_model, epoch):
    data_loader = LawDataLoader(path, batch_size, facts_dict, 'validation')
    model = LawyerPAN(lawyer_n, case_n, field_n, batch_size, device)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    model.load_state_dict(saved_model.state_dict())
    model = model.to(device)
    model.eval()

    batch_count = 0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode, labels = data_loader.next_batch()
        if labels is None:
            continue
        input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode, labels = input_lawy_ids.to(device), input_case_ids.to(device), input_field_embs.to(device), input_factions.to(device), input_pla_nums.to(device), input_def_nums.to(device), input_pla_embs.to(device), input_def_embs.to(device), facts_encode.to(device), labels.to(device)
        output = model.forward(input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode)
        output = output.view(-1)

        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    # evaluate
    accuracy = evaluation(epoch, label_all, pred_all)
    return accuracy

def evaluation(epoch, label_all, pred_all):
    pred_all = np.array(pred_all)
    pred_all2bi = np.where(pred_all > 0.5, 1, 0)
    label_all = np.array(label_all)

    accuracy = accuracy_score(label_all, pred_all2bi)
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)
    f1 = f1_score(label_all, pred_all2bi, pos_label=1, average='binary')
    precision = precision_score(label_all, pred_all2bi, pos_label=1, average='binary')
    recall = recall_score(label_all, pred_all2bi, pos_label=1, average='binary')
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n  precision= %f, recall= %f, f1_score= %f' % (epoch, accuracy, rmse, auc, precision, recall, f1))
    if epoch != 0:
        performances.append((epoch, accuracy))
    return accuracy

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

def test(epoch):
    data_loader = LawDataLoader(path, batch_size, facts_dict, 'test')
    model = LawyerPAN(lawyer_n, case_n, field_n, batch_size, device)
    print('testing model...')
    data_loader.reset()
    load_snapshot(model, model_path + 'model_epoch' + str(epoch))
    model = model.to(device)
    model.eval()

    pred_all, label_all = [], []
    while not data_loader.is_end():
        input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode, labels = data_loader.next_batch()
        if labels is None:
            continue
        input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode, labels = input_lawy_ids.to(device), input_case_ids.to(device), input_field_embs.to(device), input_factions.to(device), input_pla_nums.to(device), input_def_nums.to(device), input_pla_embs.to(device), input_def_embs.to(device), facts_encode.to(device), labels.to(device)
        output = model.forward(input_lawy_ids, input_case_ids, input_field_embs, input_factions, input_pla_nums, input_def_nums, input_pla_embs, input_def_embs, facts_encode)
        output = output.view(-1)

        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    # evaluate
    evaluation(epoch, label_all, pred_all)

def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


if __name__ == '__main__':
    train()
