import csv
import json
import pathlib


def convert_cosmos_qa() -> None:
    path = pathlib.Path('data/cosmosQA')
    convert(path=path, file='train')
    convert(path=path, file='valid')


def convert(path: pathlib.Path, file: str) -> None:
    lines = []
    with open(path / '{0}.csv'.format(file), 'r', encoding='utf-8') as fr:
        csv_reader = csv.DictReader(fr)
        for row in csv_reader:
            lines.append('{0}\n'.format(json.dumps(row, ensure_ascii=False)))
    with open(path / '{0}.jsonl'.format(file), 'w+', encoding='utf-8') as fw:
        fw.writelines(lines)


def main() -> None:
    convert_cosmos_qa()


if __name__ == '__main__':
    main()
