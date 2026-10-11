"""Train a diabetic retinopathy severity classifier."""

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision"]
# ///

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_DATASET_DIR = Path(r"D:\MiniProject\archive")
IMAGE_SIZE = 224
CLASS_COUNT = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageSample:
    image_id: str
    image_path: Path
    label: int
    patient_id: str


def read_labels(csv_path: Path, image_dir: Path) -> list[ImageSample]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    samples: list[ImageSample] = []
    missing: list[str] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as label_file:
        reader = csv.DictReader(label_file)
        if not {"image", "level"}.issubset(reader.fieldnames or []):
            raise ValueError("The label CSV must contain 'image' and 'level' columns.")
        for row in reader:
            image_id = row["image"].strip()
            label = int(row["level"])
            if label not in range(CLASS_COUNT):
                raise ValueError(f"Invalid grade {label} for image {image_id}.")
            image_path = image_dir / f"{image_id}.jpeg"
            if not image_path.is_file():
                missing.append(image_path.name)
            samples.append(
                ImageSample(
                    image_id=image_id,
                    image_path=image_path,
                    label=label,
                    patient_id=image_id.rsplit("_", maxsplit=1)[0],
                )
            )

    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} labeled images are missing; examples: {examples}")
    return samples


class RetinaDataset(Dataset):
    def __init__(self, samples: list[ImageSample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, sample.label


def build_transforms(training: bool) -> transforms.Compose:
    operations = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def split_by_patient(
    samples: list[ImageSample], fold_count: int, seed: int
) -> tuple[list[ImageSample], list[ImageSample]]:
    samples_by_patient: dict[str, list[ImageSample]] = defaultdict(list)
    total_counts = Counter(sample.label for sample in samples)
    for sample in samples:
        samples_by_patient[sample.patient_id].append(sample)

    patient_ids = list(samples_by_patient)
    random.Random(seed).shuffle(patient_ids)
    patient_ids.sort(
        key=lambda patient_id: sum(
            sum(sample.label == grade for sample in samples_by_patient[patient_id])
            / total_counts[grade]
            for grade in range(CLASS_COUNT)
        ),
        reverse=True,
    )

    fold_counts = [Counter() for _ in range(fold_count)]
    fold_patients: list[list[str]] = [[] for _ in range(fold_count)]
    fold_sizes = [0] * fold_count
    for patient_id in patient_ids:
        patient_counts = Counter(sample.label for sample in samples_by_patient[patient_id])

        def fold_score(fold_index: int) -> float:
            return sum(
                (
                    (fold_counts[fold_index][grade] + patient_counts[grade])
                    / max(total_counts[grade] / fold_count, 1)
                )
                ** 2
                for grade in range(CLASS_COUNT)
            )

        selected_fold = min(range(fold_count), key=lambda index: (fold_score(index), len(fold_patients[index])))
        fold_counts[selected_fold].update(patient_counts)
        fold_sizes[selected_fold] += len(samples_by_patient[patient_id])
        fold_patients[selected_fold].append(patient_id)

    validation_patients = set(fold_patients[0])
    train_samples = [sample for sample in samples if sample.patient_id not in validation_patients]
    validation_samples = [sample for sample in samples if sample.patient_id in validation_patients]
    return train_samples, validation_samples


def macro_f1_score(actual: list[int], predicted: list[int]) -> float:
    class_scores = []
    for grade in range(CLASS_COUNT):
        true_positive = sum(a == grade and p == grade for a, p in zip(actual, predicted))
        false_positive = sum(a != grade and p == grade for a, p in zip(actual, predicted))
        false_negative = sum(a == grade and p != grade for a, p in zip(actual, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(class_scores) / CLASS_COUNT


def quadratic_kappa(actual: list[int], predicted: list[int]) -> float:
    if not actual:
        return float("nan")
    actual_counts = Counter(actual)
    predicted_counts = Counter(predicted)
    observed_disagreement = sum(
        ((a - p) / (CLASS_COUNT - 1)) ** 2 for a, p in zip(actual, predicted)
    )
    expected_disagreement = sum(
        ((actual_grade - predicted_grade) / (CLASS_COUNT - 1)) ** 2
        * actual_counts[actual_grade]
        * predicted_counts[predicted_grade]
        / len(actual)
        for actual_grade in range(CLASS_COUNT)
        for predicted_grade in range(CLASS_COUNT)
    )
    if expected_disagreement == 0:
        return float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_dir = args.image_dir or args.dataset_dir / "resized_train" / "resized_train"
    labels_csv = args.labels_csv or args.dataset_dir / "trainLabels.csv"
    samples = read_labels(labels_csv, image_dir)
    grade_counts = Counter(sample.label for sample in samples)
    print(f"Images: {len(samples):,}")
    print(f"Grade counts: {dict(sorted(grade_counts.items()))}")

    if args.validate_only:
        print("Dataset validation passed.")
        return

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Epochs and batch size must be positive; workers cannot be negative.")

    train_samples, validation_samples = split_by_patient(samples, fold_count=5, seed=args.seed)

    train_loader = DataLoader(
        RetinaDataset(train_samples, build_transforms(training=True)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        RetinaDataset(validation_samples, build_transforms(training=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    weights = models.EfficientNet_B0_Weights.DEFAULT if not args.no_pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, CLASS_COUNT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")
    print(f"Train/validation images: {len(train_samples):,}/{len(validation_samples):,}")

    train_grade_counts = Counter(sample.label for sample in train_samples)
    class_weights = torch.tensor(
        [len(train_samples) / (CLASS_COUNT * train_grade_counts[grade]) for grade in range(CLASS_COUNT)],
        dtype=torch.float32,
        device=device,
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_kappa = float("-inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, batch_labels in train_loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_function(model(images), batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(batch_labels)

        model.eval()
        actual: list[int] = []
        predicted: list[int] = []
        with torch.no_grad():
            for images, batch_labels in validation_loader:
                logits = model(images.to(device, non_blocking=True))
                predicted.extend(logits.argmax(dim=1).cpu().tolist())
                actual.extend(batch_labels.tolist())

        accuracy = sum(
            actual_label == prediction
            for actual_label, prediction in zip(actual, predicted)
        ) / len(actual)
        macro_f1 = macro_f1_score(actual, predicted)
        kappa = quadratic_kappa(actual, predicted)
        average_loss = total_loss / len(train_samples)
        print(
            f"Epoch {epoch}/{args.epochs} | loss {average_loss:.4f} | "
            f"accuracy {accuracy:.4f} | macro-F1 {macro_f1:.4f} | QWK {kappa:.4f}"
        )

        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_count": CLASS_COUNT,
                    "image_size": IMAGE_SIZE,
                    "validation_qwk": kappa,
                },
                checkpoint_path,
            )

    print(f"Best checkpoint: {checkpoint_path} (validation QWK {best_kappa:.4f})")


if __name__ == "__main__":
    main()
