"""JYP-compatible real-noise image manifests with stable sample indices."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet canonical class IDs 0..49, in the same order as WebVision-50.
# Validate three-column ILSVRC annotations rather than trusting folder order.
ILSVRC50_SYNSETS = (
    "n01440764", "n01443537", "n01484850", "n01491361", "n01494475",
    "n01496331", "n01498041", "n01514668", "n01514859", "n01518878",
    "n01530575", "n01531178", "n01532829", "n01534433", "n01537544",
    "n01558993", "n01560419", "n01580077", "n01582220", "n01592084",
    "n01601694", "n01608432", "n01614925", "n01616318", "n01622779",
    "n01629819", "n01630670", "n01631663", "n01632458", "n01632777",
    "n01641577", "n01644373", "n01644900", "n01664065", "n01665541",
    "n01667114", "n01667778", "n01669191", "n01675722", "n01677366",
    "n01682714", "n01685808", "n01687978", "n01688243", "n01689811",
    "n01692333", "n01693334", "n01694178", "n01695060", "n01697457",
)


class IndexedImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.targets = [label for _, label in self.samples]
        self.transform = transform
        if not self.samples:
            raise ValueError("The image manifest contains no samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, index


def _require(path, expected):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Expected {expected}")
    return path


def _read_list(file_path, image_root, num_classes=50, synset_map=None):
    _require(file_path, f"annotation list {file_path.name} and image directory {image_root}")
    samples = []
    with file_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 2 and not (synset_map is not None and len(fields) == 3):
                expected = "'<relative-image-path> <class-id> [synset]'" if synset_map else "'<relative-image-path> <class-id>'"
                raise ValueError(f"Expected {expected} at {file_path}:{line_number}")
            relative, raw_label = fields[:2]
            label = int(raw_label)
            if label < 0:
                raise ValueError(f"Negative class id at {file_path}:{line_number}")
            if label >= num_classes:
                continue
            if len(fields) == 3 and fields[2] != synset_map[label]:
                raise ValueError(
                    f"ILSVRC class {label} must map to {synset_map[label]}, "
                    f"found {fields[2]} at {file_path}:{line_number}"
                )
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe image path at {file_path}:{line_number}: {relative}")
            samples.append((image_root / relative_path, label))
    if not samples:
        raise ValueError(f"No class IDs 0..{num_classes - 1} found in {file_path}")
    return samples


class Animal10N(IndexedImageDataset):
    num_classes = 10

    def __init__(self, root, split="train", transform=None):
        if split not in {"train", "test"}:
            raise ValueError("Animal-10N split must be train or test.")
        root = Path(root)
        directory = root / ("training" if split == "train" else "testing")
        _require(directory, "Animal-10N/{training,testing}/<class-digit><image-name> (JYP format)")
        samples = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            if not path.name[0].isdigit() or int(path.name[0]) >= self.num_classes:
                raise ValueError(f"Animal-10N label must be the first filename digit 0..9: {path}")
            samples.append((path, int(path.name[0])))
        super().__init__(samples, transform)
        self.root, self.split = root, split


class WebVision50(IndexedImageDataset):
    num_classes = 50

    def __init__(self, root, split="train", transform=None):
        if split not in {"train", "val"}:
            raise ValueError("WebVision-50 split must be train or val.")
        root = Path(root)
        annotation = root / "info" / ("train_filelist_google.txt" if split == "train" else "val_filelist.txt")
        image_dir = root if split == "train" else root / "val_images_256"
        _require(image_dir, f"WebVision/{'train paths from info/train_filelist_google.txt' if split == 'train' else 'val_images_256/'}")
        samples = _read_list(annotation, image_dir, num_classes=self.num_classes)
        super().__init__(samples, transform)
        self.root, self.split = root, split


class ILSVRC12_50(IndexedImageDataset):
    num_classes = 50

    def __init__(self, root, transform=None):
        root = Path(root)
        image_dir = _require(root / "ILSVRC2012_img_val", "ILSVRC2012/ILSVRC2012_img_val/ (labeled validation split)")
        annotation = root / "ILSVRC2012_val_label.txt"
        samples = _read_list(
            annotation, image_dir, num_classes=self.num_classes,
            synset_map=ILSVRC50_SYNSETS,
        )
        # JYP's annotation explicitly assigns WebVision output indices 0..49 to
        # ILSVRC validation images. Never infer these IDs from folder order.
        super().__init__(samples, transform)
        self.root, self.split = root, "test"


REAL_DATASET_META = {
    "animal10n": {"num_classes": 10, "splits": ("train", "test"), "is_real_noise": True},
    "webvision": {"num_classes": 50, "splits": ("train", "val"), "is_real_noise": True},
    "ilsvrc12_50": {"num_classes": 50, "splits": ("test",), "is_real_noise": False},
}


def load_real_base(dataset, dataset_root, split, transform=None, ilsvrc12_root=None):
    if dataset == "animal10n":
        return Animal10N(dataset_root, split, transform)
    if dataset == "webvision":
        return WebVision50(dataset_root, split, transform)
    if dataset == "ilsvrc12_50":
        return ILSVRC12_50(ilsvrc12_root or dataset_root, transform)
    raise ValueError(f"Unknown real-noise dataset: {dataset}")
