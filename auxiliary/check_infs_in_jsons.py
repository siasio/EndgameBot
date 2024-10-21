import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str)
args = parser.parse_args()


def check_infs_in_jsons(folder):
    for folder, subfolders, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(folder, file)
            if file.endswith(".json"):
                with open(filepath, "r") as f:
                    json_contents = f.read()
                if "inf" in json_contents:
                    print(f"Found inf in {filepath}")

