import argparse
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("sgf_dir", help="Path to directory with sgf files")
args = parser.parse_args()

sgfs = [args.sgf_dir] if args.sgf_dir.endswith('.sgf') else [os.path.join(args.sgf_dir, f) for f in os.listdir(args.sgf_dir) if f.endswith(".sgf")]

for sgf in tqdm(sgfs):
    with open(sgf, 'r') as f:
        sgf_content = f.read()
    no_passes = sgf_content.replace(';B[]', '').replace(';W[]', '')
    with open(sgf, 'w') as f:
        f.write(no_passes)


