"""Shared RVL-CDIP helpers for the P1 data scripts.

aharley/rvl_cdip is a loading-script dataset: one 38.7 GB gzip tar of TIFFs
plus per-split label files (data/{train,val,test}.txt, "<path> <label>" per
line). Its script yields examples in tar order, so a "dataset index" from
load_dataset is neither cheap to compute nor stable across loaders. Records
here are keyed instead by the line number in the split's label file at a
pinned repo revision, plus the image path inside the tar.
"""

import os
import random
import tarfile

DATASET_REPO = "aharley/rvl_cdip"
# Pinned repo commit; label files and tar are read at this revision.
DATASET_REVISION = "03f14a4ad0a32413eff51ca10f9f511545f2bd5b"
TAR_PATH = "data/rvl-cdip.tar.gz"
TAR_IMAGES_PREFIX = "images/"

# Label-file name for each of our split names.
SPLIT_FILES = {"train": "data/train.txt", "val": "data/val.txt", "test": "data/test.txt"}

CLASSES = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]


def hf_token():
    return os.environ.get("HF_TOKEN") or None


def download_label_file(split, revision=DATASET_REVISION):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET_REPO, SPLIT_FILES[split], repo_type="dataset",
                           revision=revision, token=hf_token())


def read_label_file(path):
    """Return [(image_path, label_int)] in file order; list position is image_index."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_path, label = line.split(" ")
            rows.append((image_path, int(label)))
    return rows


def allocate(counts, n, rng):
    """Split n across classes proportionally to counts (largest remainder).

    counts: {class: available}. Ties in the remainder are broken by rng so
    no class is favoured by its id. Never allocates more than available.
    """
    total = sum(counts.values())
    if n > total:
        raise ValueError(f"Requested {n} items but only {total} available.")
    quotas = {c: n * k / total for c, k in counts.items()}
    alloc = {c: min(int(q), counts[c]) for c, q in quotas.items()}
    order = list(counts)
    rng.shuffle(order)
    order.sort(key=lambda c: quotas[c] - int(quotas[c]), reverse=True)
    remaining = n - sum(alloc.values())
    while remaining > 0:
        for c in order:
            if remaining == 0:
                break
            if alloc[c] < counts[c]:
                alloc[c] += 1
                remaining -= 1
    return alloc


def stratified_sample(items, key, n, seed):
    """Pick n items stratified by key(item); deterministic for a given seed."""
    rng = random.Random(seed)
    by_class = {}
    for item in items:
        by_class.setdefault(key(item), []).append(item)
    classes = sorted(by_class)
    alloc = allocate({c: len(by_class[c]) for c in classes}, n, rng)
    picked = []
    for c in classes:
        picked.extend(rng.sample(by_class[c], alloc[c]))
    return picked


def iter_tar_images(revision=DATASET_REVISION, max_members=None):
    """Stream (image_path, bytes) from the RVL-CDIP tar without saving it.

    image_path is relative to the label files (the "images/" prefix is
    stripped). The tar is gzip, so this always reads from the start.
    """
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(DATASET_REPO, TAR_PATH, repo_type="dataset", revision=revision)
    token = hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for i, member in enumerate(tar):
                if max_members is not None and i >= max_members:
                    return
                if not member.isfile() or not member.name.startswith(TAR_IMAGES_PREFIX):
                    continue
                yield member.name[len(TAR_IMAGES_PREFIX):], tar.extractfile(member).read()
