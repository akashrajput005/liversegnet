import os
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset


INSTRUMENT_LABELS = ["grasper", "hook", "clipper", "scissor", "instrument", "tool"]


def _is_seg8k_split(path, split, max_vid):
    path = path.lower().replace("\\", "/")
    import re

    match = re.search(r"video(\d+)", path)
    if match:
        vid_num = int(match.group(1))
        train_max = 60
        if max_vid and max_vid <= 60:
            train_max = max(1, int(round(max_vid * 0.8)))
            if train_max >= max_vid:
                train_max = max_vid - 1 if max_vid > 1 else 1
        if split == "train":
            return vid_num <= train_max
        if split in ["val", "validation", "test"]:
            return vid_num > train_max
    return True


def _map_seg8k_mask(mask):
    mapped = np.zeros_like(mask, dtype=np.uint8)
    mapped[mask == 21] = 1
    mapped[mask == 50] = 2
    return mapped


def _get_max_video(root):
    max_vid = 0
    for p in Path(root).iterdir():
        if p.is_dir() and p.name.lower().startswith("video"):
            try:
                vid = int(p.name.lower().replace("video", ""))
                max_vid = max(max_vid, vid)
            except ValueError:
                continue
    return max_vid


def _render_instance_mask(json_path, shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    with open(json_path, "r") as f:
        data = json.load(f)
    for shape_item in data.get("shapes", []):
        label = str(shape_item.get("label", "")).lower()
        if any(x in label for x in INSTRUMENT_LABELS):
            points = np.array(shape_item["points"], dtype=np.float32)
            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, h - 1)
            pts = points.astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def _to_tensor(image, mask):
    image = image.transpose(2, 0, 1).astype("float32") / 255.0
    return torch.from_numpy(image), torch.from_numpy(mask.astype("int64"))


class Stage1Dataset(Dataset):
    def __init__(self, config, split="train", include_instances=True):
        self.split = split
        self.image_size = int(config["training"]["image_size"])
        self.samples = []
        self.transform = None
        if split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.RandomBrightnessContrast(p=0.4),
                    A.HueSaturationValue(p=0.3),
                    A.RandomGamma(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.MotionBlur(p=0.2),
                    A.GaussianBlur(p=0.2),
                    A.Sharpen(p=0.1),
                ]
            )

        seg8k_root = Path(config["data"]["cholecseg8k"])
        if seg8k_root.exists():
            max_vid = _get_max_video(seg8k_root)
            for mpath in seg8k_root.rglob("*_endo_mask.png"):
                ipath = str(mpath).replace("_mask.png", ".png")
                if os.path.exists(ipath) and _is_seg8k_split(ipath, split, max_vid):
                    self.samples.append({"image": ipath, "mask": str(mpath), "source": "seg8k"})

        if include_instances:
            inst_root = Path(config["data"]["cholecinstanceseg"])
            if inst_root.exists():
                split_folder = "val" if split in ["val", "validation"] else split
                ann_paths = list(inst_root.rglob(f"{split_folder}/**/ann_dir/*.json"))
                if not ann_paths:
                    ann_paths = list(inst_root.rglob("ann_dir/*.json"))
                img_index = self._build_image_index(inst_root)
                for jpath in ann_paths:
                    img_path = self._resolve_image_for_json(jpath, img_index)
                    if img_path:
                        self.samples.append({"image": img_path, "mask": str(jpath), "source": "instance"})

        random.shuffle(self.samples)

    def _build_image_index(self, root):
        index = {}
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in root.rglob(ext):
                if p.stem not in index:
                    index[p.stem] = str(p)
        return index

    def _resolve_image_for_json(self, json_path, index):
        json_path = Path(json_path)
        if "ann_dir" in json_path.parts:
            grandparent = json_path.parent.parent
            img_dir = grandparent / "img_dir"
            if img_dir.exists():
                candidate = img_dir / json_path.with_suffix(".png").name
                if candidate.exists():
                    return str(candidate)
                candidate = img_dir / json_path.with_suffix(".jpg").name
                if candidate.exists():
                    return str(candidate)
        return index.get(json_path.stem, None)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["image"])
        if img is None:
            raise RuntimeError(f"Failed to read image: {sample['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        if sample["source"] == "seg8k":
            mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {sample['mask']}")
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            mask = _map_seg8k_mask(mask)
            source = 0
        else:
            mask = _render_instance_mask(sample["mask"], img.shape)
            source = 1

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        image_t, mask_t = _to_tensor(img, mask)
        source_t = torch.tensor(source, dtype=torch.int64)
        return image_t, mask_t, source_t


class Stage2Dataset(Dataset):
    def __init__(self, config, split="train", crop_size=256, padding=20):
        self.split = split
        self.crop_size = crop_size
        self.padding = padding
        self.samples = []
        self.total_jsons = 0
        self.resolved_images = 0
        self.valid_masks = 0
        self.example_json = None
        self.example_image = None
        self.transform = None
        if split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.RandomBrightnessContrast(p=0.4),
                    A.HueSaturationValue(p=0.3),
                    A.RandomGamma(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.MotionBlur(p=0.2),
                    A.GaussianBlur(p=0.2),
                    A.Sharpen(p=0.1),
                ]
            )

        inst_root = Path(config["data"]["cholecinstanceseg"])
        image_root = Path(config["data"].get("cholecinstanceseg_images", inst_root))
        if inst_root.exists():
            split_folder = "val" if split in ["val", "validation"] else split
            ann_paths = list(inst_root.rglob(f"{split_folder}/**/ann_dir/*.json"))
            if not ann_paths:
                ann_paths = list(inst_root.rglob("ann_dir/*.json"))
            img_index = None
            for jpath in ann_paths:
                self.total_jsons += 1
                img_path = self._resolve_image_for_json(jpath, img_index, image_root)
                if not img_path and img_index is None:
                    img_index = self._build_image_index(image_root)
                    img_path = self._resolve_image_for_json(jpath, img_index, image_root)
                if not img_path:
                    continue
                self.resolved_images += 1
                if not self._has_instrument(jpath):
                    continue
                self.valid_masks += 1
                self.samples.append({"image": img_path, "mask": str(jpath)})
                if self.example_json is None:
                    self.example_json = str(jpath)
                    self.example_image = img_path

        random.shuffle(self.samples)

    def _build_image_index(self, root):
        cache_path = Path("logs/cholecinstanceseg_image_index.json")
        try:
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        index = {}
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in root.rglob(ext):
                if p.stem not in index:
                    index[p.stem] = str(p)
        try:
            os.makedirs(cache_path.parent, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(index, f)
        except Exception:
            pass
        return index

    def _resolve_image_for_json(self, json_path, index, image_root):
        json_path = Path(json_path)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            image_path = data.get("imagePath", None)
            if image_path:
                candidate = Path(image_path)
                if not candidate.is_absolute():
                    candidate = json_path.parent / image_path
                if candidate.exists():
                    return str(candidate)
                candidate = Path(image_root) / image_path
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass
        if "ann_dir" in json_path.parts:
            grandparent = json_path.parent.parent
            split_guess = "train" if "train" in json_path.parts else "val" if "val" in json_path.parts else "test"
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = image_root / split_guess / grandparent.name / "img_dir" / (json_path.stem + ext)
                if candidate.exists():
                    return str(candidate)
        if json_path.stem in index:
            return index[json_path.stem]
        return None

    def _has_instrument(self, json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            for shape_item in data.get("shapes", []):
                label = str(shape_item.get("label", "")).lower()
                if any(x in label for x in INSTRUMENT_LABELS):
                    return True
        except Exception:
            return False
        return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["image"])
        if img is None:
            raise RuntimeError(f"Failed to read image: {sample['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = _render_instance_mask(sample["mask"], img.shape)

        h, w = img.shape[:2]
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            cx, cy = w // 2, h // 2
            x1 = max(0, cx - self.crop_size // 2)
            y1 = max(0, cy - self.crop_size // 2)
            x2 = min(w, x1 + self.crop_size)
            y2 = min(h, y1 + self.crop_size)
        else:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w - 1, x2 + self.padding)
            y2 = min(h - 1, y2 + self.padding)

        crop = img[y1 : y2 + 1, x1 : x2 + 1]
        crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
        crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        crop_mask = cv2.resize(crop_mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=crop, mask=crop_mask)
            crop, crop_mask = augmented["image"], augmented["mask"]

        image_t, mask_t = _to_tensor(crop, crop_mask)
        return image_t, mask_t.unsqueeze(0).float()
