import json
import random

min_log = 10


def divide_data(data_path):
    '''
    1. delete lawyers who have fewer than min_log response logs
    2. divide dataset into train_set, val_set and test_set (0.7:0.1:0.2)
    :return:
    '''
    with open(data_path + 'log_data.json', encoding='utf8') as i_f:
        lawys = json.load(i_f)
    # 1. delete lawyers who have fewer than min_log response logs
    lawy_i = 0
    while lawy_i < len(lawys):
        if lawys[lawy_i]['log_num'] < min_log:
            del lawys[lawy_i]
            lawy_i -= 1
        lawy_i += 1
    # 2. divide dataset into train_set, val_set and test_set
    train_slice, val_slice, test_slice, train_set, val_set, test_set = [], [], [], [], [], []
    for lawy in lawys:
        user_id = lawy['user_id']
        lawy_train = {'user_id': user_id}
        lawy_val = {'user_id': user_id}
        lawy_test = {'user_id': user_id}
        train_size = int(lawy['log_num'] * 0.7)
        val_size = int(lawy['log_num'] * 0.1)
        test_size = lawy['log_num'] - train_size - val_size
        logs = []
        for log in lawy['logs']:
            logs.append(log)
        random.shuffle(logs)
        lawy_train['log_num'] = train_size
        lawy_train['logs'] = logs[:train_size]
        lawy_val['log_num'] = val_size
        lawy_val['logs'] = logs[train_size:train_size+val_size]
        lawy_test['log_num'] = test_size
        lawy_test['logs'] = logs[-test_size:]
        train_slice.append(lawy_train)
        val_set.append(lawy_val)
        test_set.append(lawy_test)
        # shuffle logs in train_slice together, get train_set
        for log in lawy_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'], 'knowledge_code': log['knowledge_code'],
                              'tag': log['tag'], 'member': log['member'], 'expertise': log['expertise']})
        for log in lawy_val['logs']:
            val_slice.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'], 'knowledge_code': log['knowledge_code'],
                              'tag': log['tag'], 'member': log['member'], 'expertise': log['expertise']})
        for log in lawy_test['logs']:
            test_slice.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'], 'knowledge_code': log['knowledge_code'],
                              'tag': log['tag'], 'member': log['member'], 'expertise': log['expertise']})
    random.shuffle(train_set)
    with open(data_path + 'train_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
    with open(data_path + 'val_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(val_slice, output_file, indent=4, ensure_ascii=False)
    with open(data_path + 'test_slice.json', 'w', encoding='utf8') as output_file:
        json.dump(test_slice, output_file, indent=4, ensure_ascii=False)
    with open(data_path + 'train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open(data_path + 'val_set.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set
    with open(data_path + 'test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    data_path = "./data/"
    divide_data(data_path)
