import json
import pathlib
from typing import Optional
from wikidata.client import Client
# from Wikidata.client
# from qwikidata

CHECKPOINT_LINES = 100

wd_client = Client()
# print(test.get('Q20830929'))
# print(test.get('Q20830929').state)
# print(test.get('Q20830929').type)
# print(test.get('Q20830929').attributes['labels']['en']['value'])
# print(wd_client.get('Q20830929').attributes['claims']['P31'][0]['mainsnak']['datavalue']['value']['id'])
# print(test.get('Q5').attributes['labels']['en']['value'])
BASE_PATH = pathlib.Path('data')

labels_file = BASE_PATH / 'entity_labels_en.txt'
labels = {}
types = {}
with open(labels_file, 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line_vals = line.strip().split('\t')
        # print(line_vals)
        labels.update({
            line_vals[0]: line_vals[1],
        })

types_file = BASE_PATH / 'entity_types.txt'
with open(types_file, 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line_vals = line.strip().split('\t')
        # print(line_vals)
        if line_vals[0] in types.keys():
            old_val = types.get(line_vals[0])
            old_val.append(line_vals[1])
            types.update({
                line_vals[0]: old_val,
            })
        else:
            types.update({
                line_vals[0]: [line_vals[1]],
            })

# print(labels)


def get_label(id: str) -> str:
    # return wd_client.get(id).attributes['labels']['en']['value']
    return wd_client.get(id).attributes['claims']['P31'][0]['mainsnak']['datavalue']['value']['id']


def get_name(id: str) -> str:
    # print(id)
    return wd_client.get(id).attributes['labels']['en']['value']


def convert_lit_wd_1k() -> None:
    path = pathlib.Path('data/LitWD1K')
    # print(path)
    # print((path / 'test.txt').exists())
    convert_data(path=path, file='test')
    convert_data(path=path, file='valid', new_file='dev')
    convert_data(path=path, file='train')


def convert_lit_wd_19k() -> None:
    path = pathlib.Path('data/LitWD19K')
    # print(path)
    # print((path / 'test.txt').exists())
    convert_data(path=path, file='test')
    convert_data(path=path, file='valid', new_file='dev')
    convert_data(path=path, file='train')


def convert_lit_wd_48k() -> None:
    path = pathlib.Path('data/LitWD48K')
    # print(path)
    # print((path / 'test.txt').exists())
    convert_data(path=path, file='test')
    convert_data(path=path, file='valid', new_file='dev')
    convert_data(path=path, file='train')


def convert_data(path: pathlib.Path, file: str, new_file: Optional[str] = None) -> None:
    all_lines = []
    if new_file:
        new_file_str = '{0}.json'.format(new_file)
    else:
        new_file_str = '{0}.json'.format(file)

    start_from = 0
    if (path / new_file_str).exists():
        with open(path / new_file_str, 'r') as fr:
            all_lines = json.load(fr)
            start_from = len(all_lines)
    print('starting from line {0}'.format(start_from))

    with open(path / '{0}.txt'.format(file), 'r') as fr:
        lines = fr.readlines()
        for index, line in enumerate(lines[start_from:]):
            line_vals = line.strip().split('\t')
            txt_val_1 = labels.get(line_vals[0], '')
            txt_val_2 = labels.get(line_vals[2], '')
            # print(types.get(line_vals[0], []))
            result = {
                'sent': '{0} {1} .'.format(
                    txt_val_1,
                    txt_val_2,
                ),
                'labels': [get_name(type_id) for type_id in types.get(line_vals[0], [])],
                'start': 0,
                'end': len(txt_val_1),
                'ents': [],
            }
            print(result)
            all_lines.append(result)
            if index % CHECKPOINT_LINES == 0:
                with open(path / new_file_str, 'w+') as fw:
                    json.dump(all_lines, fw)
    with open(path / new_file_str, 'w+') as fw:
        json.dump(all_lines, fw)


def main() -> None:
    convert_lit_wd_1k()
    convert_lit_wd_19k()
    convert_lit_wd_48k()
    # pass


if __name__ == '__main__':
    main()
