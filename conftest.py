import importlib

# Tests that need PyTorch + ML stack. These run locally, not in CI.
# CI only installs lightweight deps (numpy, httpx, pytest, ruff).
_TORCH_TESTS = [
    "tests/unit/test_dinov2_detector.py",
    "tests/unit/test_unet_convnext.py",
    "tests/unit/test_losses.py",
    "tests/unit/test_metrics.py",
    "tests/unit/test_evaluate.py",
    "tests/unit/test_train_dinov2.py",
    "tests/unit/test_train_unet.py",
    "tests/unit/test_dataset.py",
]

_PIL_TESTS = [
    "tests/unit/test_preprocessing.py",
    "tests/unit/test_normalization.py",
    "tests/unit/test_lung_segmentation.py",
]

# NLP evaluation metrics — sacrebleu, rouge-score, bert-score.
# Installed via requirements-eval.txt, not in CI.
_EVAL_TESTS = [
    "tests/unit/test_nlp_metrics.py",
]


def _available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


collect_ignore = []

if not _available("torch"):
    collect_ignore.extend(_TORCH_TESTS)

if not _available("PIL"):
    collect_ignore.extend(_PIL_TESTS)

if not _available("sacrebleu"):
    collect_ignore.extend(_EVAL_TESTS)