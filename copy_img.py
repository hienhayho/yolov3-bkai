import random
import shutil

with open("/mlcv2/WorkingSpace/Personal/baotg/Learn/doan/PyTorch-YOLOv3/data/a_dataset/valid.txt") as f:
    lines = f.readlines()

random.shuffle(lines)

target_path = "/mlcv2/WorkingSpace/Personal/baotg/Learn/doan/PyTorch-YOLOv3/tests"

for line in lines[:5000]:
    line = line.strip()
    shutil.copy(line, target_path)
