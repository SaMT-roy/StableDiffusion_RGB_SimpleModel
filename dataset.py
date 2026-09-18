"""
COCO 2017 captions: 64x64 images paired with real, relational English sentences
("a man riding a horse next to a fence").

The images are streamed straight from the COCO servers, resized in memory, and thrown
away - no JPEG ever touches the disk. 30,000 images at 64x64 is ~370 MB as raw bytes,
versus ~5 GB of original JPEGs.

Run `python dataset.py` once to build the cache (about 35 minutes on a home connection).
It saves progress as it goes, so you can interrupt it and start it again.
"""

import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
IMAGE_URL = "http://images.cocodataset.org/train2017/%012d.jpg"

NUM_IMAGES = 118287      # all of COCO train2017; 5 captions each
IMAGE_SIZE = 64
HOLDOUT = 2000           # the last N images are never trained on; evaluation uses them
DOWNLOAD_THREADS = 64
CHUNK = 10000            # save progress every this many images


# ----------------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------------
def _fetch(image_id):
    """Download one image and shrink it to 64x64. Returns None if anything fails."""
    try:
        raw = urllib.request.urlopen(IMAGE_URL % image_id, timeout=30).read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        # resize so the short side is 64, then crop the middle square: this keeps the
        # aspect ratio honest instead of squashing tall images into a square
        width, height = image.size
        scale = IMAGE_SIZE / min(width, height)
        image = image.resize((max(IMAGE_SIZE, round(width * scale)),
                              max(IMAGE_SIZE, round(height * scale))), Image.BICUBIC)
        left, top = (image.width - IMAGE_SIZE) // 2, (image.height - IMAGE_SIZE) // 2
        return np.asarray(image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE)))
    except Exception:
        return None


def build_cache():
    os.makedirs(DATA_DIR, exist_ok=True)
    image_path = os.path.join(DATA_DIR, "images.npy")
    caption_path = os.path.join(DATA_DIR, "captions.json")

    annotations = json.load(open(os.path.join(DATA_DIR, "captions_train2017.json")))
    captions_by_id = {}
    for row in annotations["annotations"]:
        captions_by_id.setdefault(row["image_id"], []).append(row["caption"].strip())

    # deterministic order, and only images that really do have 5 captions
    wanted = sorted(i for i in captions_by_id if len(captions_by_id[i]) >= 5)
    np.random.default_rng(0).shuffle(wanted)
    wanted = wanted[:NUM_IMAGES]

    images = list(np.load(image_path)) if os.path.exists(image_path) else []
    captions = json.load(open(caption_path)) if os.path.exists(caption_path) else []

    # Which ids we already have. Resuming by *count* would be wrong: a fetch that fails is
    # skipped, so the cache falls behind the position in `wanted`, and the next run would
    # redownload part of a finished range and misalign images against captions.
    id_path = os.path.join(DATA_DIR, "ids.json")
    if os.path.exists(id_path):
        done = set(json.load(open(id_path)))
    else:
        done = set(wanted[:len(images)])      # the original cache had no failed fetches
    remaining = [i for i in wanted if i not in done]
    print(f"{len(images)} images already cached, fetching {len(remaining)} more")

    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        for start in range(0, len(remaining), CHUNK):
            batch_ids = remaining[start:start + CHUNK]
            for image_id, array in zip(batch_ids, pool.map(_fetch, batch_ids)):
                if array is not None:
                    images.append(array)
                    captions.append(captions_by_id[image_id][:5])
                    done.add(image_id)
            np.save(image_path, np.stack(images))
            json.dump(captions, open(caption_path, "w"))
            json.dump(sorted(done), open(id_path, "w"))
            print(f"  {len(images)}/{len(wanted)} images cached", flush=True)
    print(f"done: {os.path.getsize(image_path)/1e6:.0f} MB of images")


def load():
    """
    Returns (images uint8 (N,64,64,3), captions list-of-5-sentences).

    The last HOLDOUT images are the test set: never trained on, used to check that the
    VAE reconstructs images it has not seen.
    """
    images = np.load(os.path.join(DATA_DIR, "images.npy"), mmap_mode="r")
    captions = json.load(open(os.path.join(DATA_DIR, "captions.json")))
    return images, captions


if __name__ == "__main__":
    build_cache()
    images, captions = load()
    print(f"images   {images.shape} {images.dtype}")
    print(f"captions {len(captions)} images x {len(captions[0])} sentences each")
    print(f"training on {len(images) - HOLDOUT}, holding out {HOLDOUT}")
    print("\nexample:", captions[0][0])
