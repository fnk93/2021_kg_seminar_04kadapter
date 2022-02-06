import json
import pathlib
from typing import Optional
from wikidata.client import Client
# from Wikidata.client
# from qwikidata

CHECKPOINT_LINES = 200

wd_client = Client()
# print(test.get('Q20830929'))
# print(test.get('Q20830929').state)
# print(test.get('Q20830929').type)
# print(test.get('Q20830929').attributes['labels']['en']['value'])
# print(wd_client.get('Q20830929').attributes['claims']['P31'][0]['mainsnak']['datavalue']['value']['id'])
# print(test.get('Q5').attributes['labels']['en']['value'])


def get_label(id: str) -> str:
    # return wd_client.get(id).attributes['labels']['en']['value']
    return wd_client.get(id).attributes['claims']['P31'][0]['mainsnak']['datavalue']['value']['id']


def get_name(id: str) -> str:
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
            start_from = len(json.load(fr))

    with open(path / '{0}.txt'.format(file), 'r') as fr:
        lines = fr.readlines()
        for index, line in enumerate(lines[start_from:]):
            line_vals = line.strip().split('\t')
            txt_val_1 = get_name(line_vals[0])
            txt_val_2 = get_name(line_vals[2])
            result = {
                'sent': '{0} {1} .'.format(
                    txt_val_1,
                    txt_val_2,
                ),
                'labels': [get_name(get_label(line_vals[0]))],
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


if __name__ == '__main__':
    main()
