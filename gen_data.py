from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

args.output = Path(args.output)
args.output.mkdir(exist_ok=True, parents=True)

Path(args.output / "images").mkdir(exist_ok=True, parents=True)
Path(args.output / "labels").mkdir(exist_ok=True, parents=True)

train_data = []
test_data = []

args.path = Path(args.path)

for file in tqdm(Path(args.path / "images" / "train").iterdir()):
    train_data.append(f"{args.output}/images/{file.name}")
    shutil.copy2(file, args.output / "images")

for file in tqdm(Path(args.path / "images" / "valid").iterdir()):
    test_data.append(f"{args.output}/images/{file.name}")
    shutil.copy2(file, args.output / "images")

# Copy all labels from to labels
for file in tqdm(Path(args.path / "labels" / "train").iterdir()):
    shutil.copy2(file, args.output / "labels")

for file in tqdm(Path(args.path / "labels" / "valid").iterdir()):
    shutil.copy2(file, args.output / "labels")

with open(args.output / "train.txt", "w") as f:
    for item in train_data:
        f.write("%s\n" % item)

with open(args.output / "test.txt", "w") as f:
    for item in test_data:
        f.write("%s\n" % item)
