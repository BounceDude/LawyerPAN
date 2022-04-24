import configparser

def get_config(path, section):
    config = configparser.ConfigParser()
    config.read(path)
    if section == "data":
        user_num = int(config.get("data", "user_num"))
        item_num = int(config.get("data", "item_num"))
        field_num = int(config.get("data", "field_num"))
        user_num += 1
        return user_num, item_num, field_num
    else:
        raise RuntimeError('Config Attr Error!')
