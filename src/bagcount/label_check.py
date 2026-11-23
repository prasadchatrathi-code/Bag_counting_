"""Pre-training label-count and class-balance verification.

Used by ``mise run check-balance`` and automatically by ``mise run train``
before any weights are loaded or training starts. Reports the number of
classes, the box count for every class, and the differences between classes
and between train/val. A failed check exits non-zero so it can gate training.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MIN_TRAIN = 100
DEFAULT_MIN_VAL = 10
DEFAULT_MIN_CLASS_FRACTION = 0.05
DEFAULT_MAX_SPLIT_RATIO = 3.0
DEFAULT_MAX_CLASS_RATIO = 15.0
DEFAULT_CLASS = "bag"


@dataclass
class SplitCounts:
    """Box counts for one dataset split."""

    split: str
    label_files: int
    labelled_images: int
    per_class: dict[int, int]
    names: dict[int, str]
    total: int = 0

    def count(self, class_id: int) -> int:
        return self.per_class.get(class_id, 0)

    def fraction(self, class_id: int) -> float:
        return self.count(class_id) / self.total if self.total else 0.0

    def density(self, class_id: int) -> float:
        return self.count(class_id) / self.label_files if self.label_files else 0.0

    def label(self, class_id: int) -> str:
        return self.names.get(class_id, f"class{class_id}")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True  # non-fatal checks are reported as warnings


def _ratio(a: float, b: float) -> float:
    if b == 0:
        return float("inf") if a > 0 else 1.0
    return a / b


def _fmt_ratio(r: float) -> str:
    return "inf" if r == float("inf") else f"{r:.2f}x"


@dataclass
class Verification:
    """Result of the pre-training checks, including the full class breakdown."""

    data_yaml: Path
    class_id: int
    class_name: str
    num_classes: int = 0
    names: dict[int, str] = field(default_factory=dict)
    class_ids: list[int] = field(default_factory=list)
    counts: dict[str, SplitCounts] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.fatal)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.fatal]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.fatal]

    def class_counts(self) -> dict[int, dict[str, int]]:
        """Return ``{class_id: {split: count}}`` for every declared/seen class."""
        return {
            class_id: {split: s.count(class_id) for split, s in self.counts.items()}
            for class_id in self.class_ids
        }

    def render(self) -> str:
        lines = [
            f"Label balance check: {self.data_yaml} "
            f"(target class {self.class_id} = {self.class_name})",
            f"Classes: {self.num_classes} ("
            + ", ".join(f"{i}={n}" for i, n in sorted(self.names.items())) + ")",
        ]
        splits = list(self.counts)
        width = max(
            [len(self.names.get(i, f"class{i}")) for i in self.class_ids] + [5, 10]
        )

        if self.counts:
            header = f"  {'class':<{width}} " + " ".join(f"{s:>8}" for s in splits)
            if len(splits) >= 2:
                header += f"  {'diff':>8}  {'split ratio':>11}"
            lines += ["", "Per-class counts:", header]
            for class_id in self.class_ids:
                label = self.names.get(class_id, f"class{class_id} (undeclared)")
                values = [s.count(class_id) for s in self.counts.values()]
                row = f"  {label:<{width}} " + " ".join(f"{v:>8}" for v in values)
                if len(splits) >= 2:
                    row += f"  {values[0] - values[1]:>+8}  "
                    row += f"{_fmt_ratio(_ratio(values[0], values[1])):>11}"
                lines.append(row)
            totals = [s.total for s in self.counts.values()]
            row = f"  {'total':<{width}} " + " ".join(f"{v:>8}" for v in totals)
            if len(splits) >= 2:
                row += f"  {totals[0] - totals[1]:>+8}  "
                row += f"{_fmt_ratio(_ratio(totals[0], totals[1])):>11}"
            lines.append(row)

            lines += ["", "Class differences within each split:"]
            for split, s in self.counts.items():
                ranked = sorted(
                    ((i, s.count(i)) for i in self.class_ids), key=lambda x: -x[1]
                )
                hi_id, hi = ranked[0]
                lo_id, lo = ranked[-1]
                lines.append(
                    f"  {split:<5}: most={s.label(hi_id)} {hi}, "
                    f"least={s.label(lo_id)} {lo}, diff={hi - lo}, "
                    f"ratio={_fmt_ratio(_ratio(hi, lo))}"
                )

            lines += ["", "Target-class detail:"]
            for split, s in self.counts.items():
                lines.append(
                    f"  {split:<5}: label_files={s.label_files} labelled={s.labelled_images} "
                    f"{self.class_name}={s.count(self.class_id)} "
                    f"total={s.total} share={s.fraction(self.class_id):.1%} "
                    f"density={s.density(self.class_id):.2f}/img"
                )

        lines += ["", "Checks:"]
        for c in self.checks:
            if c.ok:
                status = "PASS"
            elif c.fatal:
                status = "FAIL"
            else:
                status = "WARN"
            lines.append(f"  [{status}] {c.name} — {c.detail}")
        if self.ok:
            suffix = ""
            if self.warnings:
                suffix = f" with {len(self.warnings)} warning(s)"
            lines.append(f"RESULT: PASS ({len(self.checks)} checks){suffix} — ready for training")
        else:
            lines.append(
                f"RESULT: FAIL ({len(self.failures)} of {len(self.checks)} checks failed) "
                "— fix the labels before training"
            )
        return "\n".join(lines)


def _parse_names(text: str) -> dict[int, str]:
    """Minimal fallback parser for a YOLO ``names:`` map (no PyYAML needed)."""
    names: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        if re.match(r"^names\s*:", line):
            inline = line.split(":", 1)[1].strip()
            if inline.startswith("{"):
                for item in inline.strip("{}").split(","):
                    if ":" in item:
                        k, v = item.split(":", 1)
                        try:
                            names[int(k.strip())] = v.strip().strip("'\"")
                        except ValueError:
                            pass
            in_names = True
            continue
        if in_names and re.match(r"^\s+\S", line):
            m = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if m:
                names[int(m.group(1))] = m.group(2).strip().strip("'\"")
        elif in_names and line.strip():
            in_names = False
    return names


def read_class_names(data_yaml: Path) -> dict[int, str]:
    """Read ``names`` from a YOLO data.yaml as ``{id: name}``."""
    text = Path(data_yaml).read_text()
    raw: object
    try:
        import yaml  # installed transitively with rfdetr/ultralytics

        raw = (yaml.safe_load(text) or {}).get("names") or {}
    except ImportError:
        raw = _parse_names(text)
    if isinstance(raw, (list, tuple)):
        return {i: str(name) for i, name in enumerate(raw)}
    return {int(k): str(v) for k, v in dict(raw).items()}


def resolve_class(names: dict[int, str], class_spec: str | int) -> int:
    """Resolve a class id or (case-insensitive) class name to its id."""
    spec = str(class_spec).strip().lower()
    for class_id, name in names.items():
        if spec == str(class_id) or spec == str(name).lower():
            return class_id
    available = ", ".join(f"{i}={n}" for i, n in sorted(names.items()))
    raise ValueError(f"unknown class {class_spec!r}; available classes: {available}")


def count_split(root: Path, split: str, names: dict[int, str]) -> SplitCounts:
    """Count labelled boxes in ``labels/<split>/*.txt``."""
    label_dir = Path(root) / "labels" / split
    files = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
    per_class: dict[int, int] = defaultdict(int)
    labelled = 0
    for f in files:
        has_box = False
        for line in f.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            per_class[class_id] += 1
            has_box = True
        if has_box:
            labelled += 1
    return SplitCounts(
        split=split,
        label_files=len(files),
        labelled_images=labelled,
        per_class=dict(per_class),
        names=names,
        total=sum(per_class.values()),
    )


def verify_labels(
    data_yaml: Path,
    *,
    class_spec: str | int = DEFAULT_CLASS,
    min_train: int = DEFAULT_MIN_TRAIN,
    min_val: int = DEFAULT_MIN_VAL,
    min_class_fraction: float = DEFAULT_MIN_CLASS_FRACTION,
    max_split_ratio: float = DEFAULT_MAX_SPLIT_RATIO,
    max_class_ratio: float = DEFAULT_MAX_CLASS_RATIO,
    splits: tuple[str, ...] = ("train", "val"),
) -> Verification:
    """Verify counts and balance for one class and report every class.

    Fails (``Verification.ok is False``) when a split has no labels, the target
    class has too few boxes, its share of all boxes is too small, the most and
    least frequent classes are too far apart, or the target density differs too
    much between train and val. Declared classes with no boxes are warnings.
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset config not found: {data_yaml}")
    names = read_class_names(data_yaml)
    class_id = resolve_class(names, class_spec)
    result = Verification(
        data_yaml=data_yaml,
        class_id=class_id,
        class_name=names[class_id],
        num_classes=len(names),
        names=names,
    )

    for split in splits:
        s = count_split(data_yaml.parent, split, names)
        result.counts[split] = s
        minimum = min_train if split == "train" else min_val
        result.checks.append(
            Check(f"{split} label files", s.label_files > 0,
                  f"{s.label_files} files ({s.labelled_images} with boxes)")
        )
        result.checks.append(
            Check(f"{split} {result.class_name} boxes", s.count(class_id) >= minimum,
                  f"{s.count(class_id)} >= {minimum}")
        )
        result.checks.append(
            Check(f"{split} {result.class_name} share",
                  s.fraction(class_id) >= min_class_fraction,
                  f"{s.fraction(class_id):.1%} >= {min_class_fraction:.1%} "
                  f"of {s.total} boxes")
        )
        present = [(i, s.count(i)) for i in names if s.count(i) > 0]
        if present:
            ranked = sorted(present, key=lambda x: -x[1])
            hi_id, hi = ranked[0]
            lo_id, lo = ranked[-1]
            class_ratio = _ratio(hi, lo)
            detail = (
                f"most={s.label(hi_id)} {hi}, least={s.label(lo_id)} {lo}, "
                f"diff={hi - lo}, ratio={_fmt_ratio(class_ratio)} "
                f"<= {max_class_ratio:.2f}x"
            )
        else:
            class_ratio = float("inf")
            detail = "no labelled boxes in this split"
        result.checks.append(
            Check(f"{split} class balance", class_ratio <= max_class_ratio, detail)
        )

    observed = {i for s in result.counts.values() for i in s.per_class}
    result.class_ids = sorted(set(names) | observed)
    missing = [
        f"{i}={n}" for i, n in sorted(names.items())
        if any(s.count(i) == 0 for s in result.counts.values())
    ]
    result.checks.append(
        Check("all declared classes labelled in both splits", not missing,
              "none missing" if not missing else ", ".join(missing), fatal=False)
    )

    if len(result.counts) >= 2:
        densities = [s.density(class_id) for s in result.counts.values()]
        lo, hi = min(densities), max(densities)
        ratio = _ratio(hi, lo)
        detail = f"{_fmt_ratio(ratio)} <= {max_split_ratio:.2f}x "
        detail += "(" + ", ".join(
            f"{k}={s.density(class_id):.2f}/img" for k, s in result.counts.items()
        ) + ")"
        result.checks.append(
            Check(f"{result.class_name} train/val density balance",
                  ratio <= max_split_ratio, detail)
        )
    return result
