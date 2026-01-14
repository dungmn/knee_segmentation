from pathlib import Path
from pycocotools.coco import COCO
import numpy as np
import cv2
from rich import print
from rich.progress import track


# Convert COCO -> Mask
def convert_coco_to_mask(src_root,out_root):
    src_root = Path(src_root)
    out_root = Path(out_root)
    ann_file = src_root / "annotations" / "instances_default.json"
    out_root.mkdir(exist_ok=True)

    coco = COCO(ann_file)
    categories = coco.loadCats(coco.getCatIds())
    class_map = {cat['id']: cat['id'] for cat in categories}
    # print(f"[green]Class Map:[/green] {class_map}")

    for img_id in track(coco.getImgIds(), description=f"Converting {src_root.name}"):
        img_info = coco.loadImgs(img_id)[0]
        image = cv2.imread(str(src_root / "images" / img_info['file_name']))
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            cat_id = ann['category_id']
            mask_seg = coco.annToMask(ann)
            mask[mask_seg == 1] = class_map[cat_id]

        stem = f"{Path(img_info['file_name']).stem}"
        # cv2.imwrite(str(out_img / f"{stem}.png"), image)
        cv2.imwrite(str(out_root / f"{stem}.png"), mask)
    
    print(f"[blue]Masks saved to:[/blue] {out_root}")


if __name__ == "__main__":

    input_dir = "data/processed/annotations/post-trans-27-random-flipped/batch_000"
    convert_coco_to_mask(input_dir)