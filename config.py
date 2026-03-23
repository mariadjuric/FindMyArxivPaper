from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"

DATA_PATH = RAW_DIR / "papers.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
PROCESSED_CSV_PATH = PROCESSED_DIR / "papers_processed.csv"

TEXT_COLUMN = "abstract"
LABEL_COLUMN = "category"
TITLE_COLUMN = "title"
MODEL_TEXT_COLUMN = "combined_text"
ID_COLUMN = "paper_id"
YEAR_COLUMN = "year"

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEST_SIZE = 0.25
RANDOM_STATE = 42
TOP_K = 5


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
