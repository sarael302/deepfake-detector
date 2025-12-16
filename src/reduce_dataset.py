import os
import random
import shutil
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def reduce_dataset(input_dir, output_dir, max_per_class=2000):
    """
    Reduce dataset size by keeping only a fixed number of images per class
    for each split: train / validation / test.

    input_dir : dataset_split/
    output_dir : dataset_small/
    max_per_class : max number of images to keep per class per split
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"]

    for split in splits:
        split_dir = input_dir / split
        classes = [d.name for d in split_dir.iterdir() if d.is_dir()]

        for cls in classes:
            src_cls = split_dir / cls
            dst_cls = output_dir / split / cls
            dst_cls.mkdir(parents=True, exist_ok=True)

            images = list(src_cls.glob("*"))
            random.shuffle(images)

            selected = images[:max_per_class]

            for img in selected:
                shutil.copy2(img, dst_cls / img.name)

            print(f"[OK] {split}/{cls} -> {len(selected)} images copied")

    print("\nDataset successfully reduced!")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="dataset_split directory")
    parser.add_argument("--out", default="./dataset_small", help="Output directory")
    parser.add_argument("--max", type=int, default=2000, help="Images per class per split")
    args = parser.parse_args()

    reduce_dataset(args.src, args.out, max_per_class=args.max)
