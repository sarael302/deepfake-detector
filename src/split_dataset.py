import os
import shutil
import random

def make_small_dataset(src_dir, out_dir, percentage=0.2):
    """Create a smaller dataset keeping the same structure train/test/validation."""

    splits = ["train", "test", "validation"]
    classes = ["Fake", "Real"]

    for split in splits:
        for cls in classes:

            src_class_dir = os.path.join(src_dir, split, cls)
            out_class_dir = os.path.join(out_dir, split, cls)

            # Create output directory
            os.makedirs(out_class_dir, exist_ok=True)

            images = os.listdir(src_class_dir)
            random.shuffle(images)

            # Calculate how many images to take
            keep_count = int(len(images) * percentage)
            selected = images[:keep_count]

            print(f"[{split}/{cls}] keeping {keep_count}/{len(images)} images")

            # Copy the selected files
            for img in selected:
                shutil.copy(
                    os.path.join(src_class_dir, img),
                    os.path.join(out_class_dir, img)
                )


if __name__ == "__main__":
    make_small_dataset(
        src_dir="dataset_small",
        out_dir="dataset_tiny",
        percentage=0.2   # <= change percentage here
    )
