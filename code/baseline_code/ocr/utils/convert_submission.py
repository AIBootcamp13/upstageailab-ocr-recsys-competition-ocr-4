import json
import argparse
import pandas as pd
from pathlib import Path


def convert_json_to_csv(json_path, output_path):
    csv_file = Path(output_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    assert 'images' in data, "The JSON file doesn't contain the 'images' key."

    rows = []
    for filename, content in data['images'].items():
        assert 'words' in content, f"The '{filename}' doesn't contain the 'words' key."

        polygons = []
        for idx, word in content['words'].items():
            assert 'points' in word, f"'{idx}' in '{filename}' doesn't contain the 'points' key."

            points = word['points']
            assert len(points) > 0, f"No points found in '{idx}' of '{filename}'."

            polygon = ' '.join([' '.join(map(str, point)) for point in points])
            polygons.append(polygon)

        polygons_str = '|'.join(polygons)
        rows.append([filename, polygons_str])

    df = pd.DataFrame(rows, columns=['filename', 'polygons'])
    df.to_csv(output_path, index=False)

    return len(rows), str(csv_file.resolve())


def convert():
    parser = argparse.ArgumentParser(description='Convert JSON to CSV')
    parser.add_argument('-J', '--json_path', type=str, required=True,
                        help='Path to the input JSON file')
    parser.add_argument('-O', '--output_path', type=str, required=True,
                        help='Path to the output CSV file')

    args = parser.parse_args()

    result = convert_json_to_csv(args.json_path, args.output_path)
    if result:
        num_rows, output_file = result
        print(f"Successfully converted {num_rows} rows to '{output_file}'")


if __name__ == "__main__":
    convert()
