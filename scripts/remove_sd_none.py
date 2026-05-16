import argparse
import json
import os
import tempfile


def filter_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as src, open(
        output_path, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("short description") == "nan":
                continue
            dst.write(line)


def main():
    parser = argparse.ArgumentParser(
        description="Remove jsonl lines where short description is nan."
    )
    parser.add_argument("input", help="Input jsonl file path")
    parser.add_argument(
        "--output",
        help="Output jsonl file path. If omitted, input file is overwritten.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if output_path is None:
        dirpath = os.path.dirname(input_path)
        fd, temp_path = tempfile.mkstemp(
            prefix="tmp_", suffix=".jsonl", dir=dirpath or None
        )
        os.close(fd)
        try:
            filter_jsonl(input_path, temp_path)
            os.replace(temp_path, input_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    else:
        filter_jsonl(input_path, output_path)


if __name__ == "__main__":
    main()
