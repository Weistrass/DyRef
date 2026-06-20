import json
import random
import os
from PIL import Image

IMAGE_ROOT = "/path/to/your/datasets"
MAX_RATIO = 8.0


def _full_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(IMAGE_ROOT, p)


def _is_extreme_ratio(p: str) -> bool:
    try:
        with Image.open(_full_path(p)) as img:
            w, h = img.size
        if w == 0 or h == 0:
            return True
        ratio = max(w / h, h / w)
        return ratio > MAX_RATIO
    except (OSError, ValueError, RuntimeError):
        # 打不开或异常，直接过滤
        return True


def _has_extreme_images(paths) -> bool:
    if paths is None:
        return False
    if isinstance(paths, str):
        return _is_extreme_ratio(paths)
    if isinstance(paths, list):
        return any(_is_extreme_ratio(p) for p in paths)
    return False


with open('/path/to/your/train_data.json', 'r') as f:
    data = json.load(f)

for item in data:
    if len(item['edit_image']) >= 5:
        continue
    if _has_extreme_images(item.get('edit_image')) or _has_extreme_images(item.get('image')):
        continue
    is_style = False
    for edit_image in item['edit_image']:
        if 'style/reference' in edit_image:
            is_style = True
            break
    if is_style:
        item['is_style'] = True
        is_test = random.random() > 0.8
        if is_test:
            item['is_test'] = True
            with open('./ours/test.jsonl', 'a') as f:
                f.write(json.dumps(item) + '\n')
        else:
            item['is_test'] = False
            with open('./ours/train.jsonl', 'a') as f:
                f.write(json.dumps(item) + '\n')
