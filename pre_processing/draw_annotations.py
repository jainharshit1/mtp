import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "DATASET" / "RFDETR_DATASET"
OUTPUT_ROOT = DATASET_ROOT / "visualized"
SPLITS = ("train", "valid", "test")
SAMPLE_LIMIT = None


def color_for(category_id):
	colors = [(255, 70, 70), (70, 180, 255), (80, 210, 100), (255, 190, 60), (210, 100, 255)]
	return colors[category_id % len(colors)]


def main():
	font = ImageFont.load_default()
	for split in SPLITS:
		split_root = DATASET_ROOT / split
		annotation_path = split_root / "_annotations.coco.json"
		output_root = OUTPUT_ROOT / split
		data = json.loads(annotation_path.read_text(encoding="utf-8"))
		images = {item["id"]: item for item in data["images"]}
		categories = {item["id"]: item["name"] for item in data["categories"]}
		annotations_by_image = {image_id: [] for image_id in images}
		for annotation in data["annotations"]:
			annotations_by_image[annotation["image_id"]].append(annotation)

		output_root.mkdir(parents=True, exist_ok=True)
		records = list(images.values())
		if SAMPLE_LIMIT is not None:
			records = records[:SAMPLE_LIMIT]

		for image_record in records:
			image_path = split_root / image_record["file_name"]
			with Image.open(image_path).convert("RGB") as image:
				draw = ImageDraw.Draw(image)
				for annotation in annotations_by_image[image_record["id"]]:
					x, y, width, height = annotation["bbox"]
					x2, y2 = x + width, y + height
					category_id = annotation["category_id"]
					label = f"{category_id}: {categories[category_id]}"
					color = color_for(category_id)
					draw.rectangle((x, y, x2, y2), outline=color, width=3)
					text_y = max(0, y - 16)
					text_box = draw.textbbox((x, text_y), label, font=font)
					draw.rectangle(text_box, fill=color)
					draw.text((x, text_y), label, fill="black", font=font)
				output_path = output_root / image_record["file_name"]
				image.save(output_path)

		print(f"{split}: rendered {len(records)} images to {output_root}")


if __name__ == "__main__":
	main()
