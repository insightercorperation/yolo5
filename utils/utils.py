import json

def json_load(json_path):
    with open(json_path, encoding='UTF-8') as json_file:
        json_data = json.load(json_file)
    return json_data

def json_save(json_path, data):
    f = open(json_path, 'w', encoding='UTF-8')
    json.dump(data, f, ensure_ascii=False)