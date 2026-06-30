
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
 
EXPECTED_DIRS = [
    "data",
    "models",
    "training",
    "report_generation",
    "evaluation",
    "api",
    "deployment",
    "monitoring",
    "tests",
    "docs",
    "configs",
]
 
 
def test_project_root_has_expected_top_level_dirs():
    for dirname in EXPECTED_DIRS:
        assert (PROJECT_ROOT / dirname).is_dir(), f"missing expected directory: {dirname}"
 
 
def test_readme_exists():
    assert (PROJECT_ROOT / "README.md").is_file()
 
 
def test_medical_disclaimer_exists():
    assert (PROJECT_ROOT / "docs" / "medical_disclaimer.md").is_file()
 