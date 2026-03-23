from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
SITE_DIR = OUTPUT_DIR / "site"

DATA_PATH = RAW_DIR / "papers.csv"
ARXIV_DATA_PATH = RAW_DIR / "arxiv_astro_ph_papers.csv"
PERFECT_DATA_PATH = RAW_DIR / "papers_perfect.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
PROCESSED_CSV_PATH = PROCESSED_DIR / "papers_processed.csv"

TEXT_COLUMN = "abstract"
LABEL_COLUMN = "category"
TITLE_COLUMN = "title"
MODEL_TEXT_COLUMN = "combined_text"
ID_COLUMN = "paper_id"
YEAR_COLUMN = "year"
AUTHORS_COLUMN = "authors"
PRIMARY_CATEGORY_COLUMN = "primary_category"
PUBLISHED_COLUMN = "published"
UPDATED_COLUMN = "updated"
URL_COLUMN = "url"

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEST_SIZE = 0.25
RANDOM_STATE = 42
TOP_K = 5

DEFAULT_ARXIV_CATEGORIES = [
    "astro-ph.GA",
    "astro-ph.SR",
    "astro-ph.HE",
    "astro-ph.CO",
    "astro-ph.EP",
    "astro-ph.IM",
]

CATEGORY_COLORS = {
    "astro-ph.GA": "#3b82f6",
    "astro-ph.SR": "#f97316",
    "astro-ph.HE": "#22c55e",
    "astro-ph.CO": "#ef4444",
    "astro-ph.EP": "#a855f7",
    "astro-ph.IM": "#a16207",
}


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, SITE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
