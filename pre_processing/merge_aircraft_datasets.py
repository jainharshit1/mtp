import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "DATASET"

# Keep class distinctions by default. Set to False to merge every annotation into
# one class named "defect".
PRESERVE_CLASSES = True

SOURCES = [
    {
        "name": "aircraft-dataset",
        "root": PROJECT_ROOT / "aircraft-dataset" / "content" / "Aircraft_dataset",
        "format": "kaggle_inline",
        "splits": {"train": "Aircraft_train.json", "val": "Aircraft_val.json", "test": "Aircraft_test.json"},
    },
    {
        "name": "Aircraft-AI-Dataset-4",
        "root": PROJECT_ROOT / "Aircraft-AI-Dataset-4",
        "format": "roboflow_coco",
        "splits": {"train": "train/_annotations.coco.json", "val": "valid/_annotations.coco.json", "test": "test/_annotations.coco.json"},
    },
]


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def add_category(category_name, categories, category_ids):
    normalized_name = category_name.strip().lower()
    if normalized_name not in category_ids:
        category_ids[normalized_name] = len(category_ids) + 1
        categories.append({"id": category_ids[normalized_name], "name": normalized_name, "supercategory": "defect"})
    return category_ids[normalized_name]


def copy_image(source_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def locate_kaggle_image(root, split, file_name):
    split_directory = {"train": "train", "val": "val", "test": "test"}[split]
    image_path = root / "images" / split_directory / file_name
    if image_path.exists():
        return image_path
    raise FileNotFoundError(f"Kaggle image not found: {image_path}")


def convert_kaggle_split(source, split, categories, category_ids):
    root = source["root"]
    data = read_json(root / source["splits"][split])
    images = []
    annotations = []
    next_image_id = 1
    next_annotation_id = 1

    for item in data["images"]:
        file_name = item["file_name"]
        source_image = locate_kaggle_image(root, split, file_name)
        with Image.open(source_image) as image:
            width, height = image.size

        output_name = f"aircraft-dataset__{file_name}"
        copy_image(source_image, OUTPUT_ROOT / "images" / split / output_name)
        image_id = next_image_id
        next_image_id += 1
        images.append({"id": image_id, "file_name": output_name, "width": width, "height": height})

        for item_annotation in item.get("annotations", []):
            box = item_annotation["bounding_box_normalized"]
            box_width = box["width"] * width
            box_height = box["height"] * height
            x = (box["x_center"] * width) - (box_width / 2)
            y = (box["y_center"] * height) - (box_height / 2)
            category_name = item_annotation.get("category_name", "defect") if PRESERVE_CLASSES else "defect"
            annotations.append({
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": add_category(category_name, categories, category_ids),
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
                "segmentation": [],
                "iscrowd": 0,
            })
            next_annotation_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def convert_roboflow_split(source, split, categories, category_ids):
    root = source["root"]
    data = read_json(root / source["splits"][split])
    images = []
    annotations = []
    image_id_map = {}

    for old_image in data.get("images", []):
        old_file_name = old_image["file_name"]
        source_image = root / {"train": "train", "val": "valid", "test": "test"}[split] / old_file_name
        if not source_image.exists():
            raise FileNotFoundError(f"Roboflow image not found: {source_image}")
        output_name = f"Aircraft-AI-Dataset-4__{old_file_name}"
        copy_image(source_image, OUTPUT_ROOT / "images" / split / output_name)
        new_image_id = len(images) + 1
        image_id_map[old_image["id"]] = new_image_id
        images.append({
            "id": new_image_id,
            "file_name": output_name,
            "width": old_image["width"],
            "height": old_image["height"],
        })

    for old_annotation in data.get("annotations", []):
        old_category = next(category for category in data["categories"] if category["id"] == old_annotation["category_id"])
        category_name = old_category["name"] if PRESERVE_CLASSES else "defect"
        bbox = old_annotation["bbox"]
        annotations.append({
            "id": len(annotations) + 1,
            "image_id": image_id_map[old_annotation["image_id"]],
            "category_id": add_category(category_name, categories, category_ids),
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "segmentation": old_annotation.get("segmentation", []),
            "iscrowd": old_annotation.get("iscrowd", 0),
        })

    return {"images": images, "annotations": annotations, "categories": categories}


def merge_split(split):
    categories = []
    category_ids = {}
    merged_images = []
    merged_annotations = []
    next_image_id = 1
    next_annotation_id = 1

    for source in SOURCES:
        converter = convert_kaggle_split if source["format"] == "kaggle_inline" else convert_roboflow_split
        converted = converter(source, split, categories, category_ids)
        image_id_map = {}
        for image in converted["images"]:
            old_id = image["id"]
            image["id"] = next_image_id
            image_id_map[old_id] = next_image_id
            next_image_id += 1
            merged_images.append(image)
        for annotation in converted["annotations"]:
            annotation["id"] = next_annotation_id
            annotation["image_id"] = image_id_map[annotation["image_id"]]
            next_annotation_id += 1
            merged_annotations.append(annotation)

    output = {
        "info": {"description": "Merged aircraft defect datasets", "preserve_classes": PRESERVE_CLASSES},
        "licenses": [],
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories,
    }
    output_path = OUTPUT_ROOT / "annotations" / f"instances_{split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"{split}: {len(merged_images)} images, {len(merged_annotations)} annotations -> {output_path}")


def main():
    for split in ("train", "val", "test"):
        merge_split(split)
    print(f"Classes preserved: {PRESERVE_CLASSES}")
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
