import json
import pathlib
from typing import Optional
from wikidata.client import Client
# from Wikidata.client
# from qwikidata

CHECKPOINT_LINES = 1000

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

cached_names = {}
cached_names_file = BASE_PATH / 'cached_names.json'
if cached_names_file.exists():
    with open(cached_names_file, 'r', encoding='utf-8') as fr:
        cached_names = json.load(fr)

# print(labels)


def save_cached_names() -> None:
    with open(cached_names_file, 'w+', encoding='utf-8') as fw:
        json.dump(cached_names, fw, ensure_ascii=False)


def get_label(id: str) -> str:
    # return wd_client.get(id).attributes['labels']['en']['value']
    # if id in cached_names.keys():
    #     return cached_names[id]
    # else:
    label = wd_client.get(id).attributes['claims']['P31'][0]['mainsnak']['datavalue']['value']['id']
    # cached_names[id] = label
    return label


def get_name(id: str) -> Optional[str]:
    # print(id)
    if id in cached_names.keys():
        return cached_names[id]
    else:
        attributes = wd_client.get(id).attributes
        # print(attributes)
        try:
            label_name = attributes['labels']['en']['value']
            cached_names[id] = label_name
            return label_name
        except KeyError:
            return None


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
        with open(path / new_file_str, 'r', encoding='utf-8') as fr:
            all_lines = json.load(fr)
            start_from = len(all_lines) + 1
    print('starting from line {0}'.format(start_from))

    with open(path / '{0}.txt'.format(file), 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for index, line in enumerate(lines[start_from:]):
        print('Line {0}/{1}'.format(
            start_from + index,
            len(lines),
        ))
        line_vals = line.strip().split('\t')
        txt_val_1 = labels.get(line_vals[0], '')
        txt_val_2 = labels.get(line_vals[2], '')
        # print(types.get(line_vals[0], []))
        labels_pre = [get_name(type_id) for type_id in types.get(line_vals[0], [])]
        labels_list = [x for x in labels_pre if x is not None]
        if len(labels_list) <= 0:
            continue
        result = {
            'sent': '{0} {1} .'.format(
                txt_val_1,
                txt_val_2,
            ),
            'labels': labels_list,
            'start': 0,
            'end': len(txt_val_1),
            'ents': [],
        }
        # print(result)
        all_lines.append(result)
        if index % CHECKPOINT_LINES == 0:
            with open(path / new_file_str, 'w+', encoding='utf-8') as fw:
                json.dump(all_lines, fw, ensure_ascii=False)
    with open(path / new_file_str, 'w+', encoding='utf-8') as fw:
        json.dump(all_lines, fw, ensure_ascii=False)


def main() -> None:
    try:
        convert_lit_wd_1k()
        convert_lit_wd_19k()
        convert_lit_wd_48k()
    except Exception as exc:
        print(exc)
    finally:
        save_cached_names()
    # pass


if __name__ == '__main__':
    main()
