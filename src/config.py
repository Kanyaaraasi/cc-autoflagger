from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
CSV_DIR = DATASET_DIR / "csv"
JSON_DIR = DATASET_DIR / "json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_CSV = CSV_DIR / "hackathon_train.csv"
VAL_CSV = CSV_DIR / "hackathon_val.csv"
TEST_CSV = CSV_DIR / "hackathon_test.csv"

# --- Target ---
TARGET = "has_ticket"

# --- Columns that MUST NOT be used as features (data leakage) ---
LEAKAGE_COLS = [
    "has_ticket",
    "ticket_has_reason",
    "ticket_priority",
    "ticket_status",
    "ticket_initial_notes",
    "ticket_resolution_notes",
    "ticket_cat_audio_issue",
    "ticket_cat_audio_notes",
    "ticket_cat_elevenlabs",
    "ticket_cat_elevenlabs_notes",
    "ticket_cat_openai",
    "ticket_cat_openai_notes",
    "ticket_cat_supabase",
    "ticket_cat_supabase_notes",
    "ticket_cat_scheduler_aws",
    "ticket_cat_scheduler_aws_notes",
    "ticket_cat_other",
    "ticket_cat_other_notes",
    "ticket_raised_at",
    "ticket_resolved_at",
]

# --- ID / non-feature columns ---
ID_COLS = ["call_id"]

# --- Text columns (processed separately) ---
TEXT_COLS = [
    "transcript_text",
    "validation_notes",
    "responses_json",
    "whisper_transcript",
]

# --- Columns to drop before feature matrix (IDs + leakage + raw text + timestamps) ---
DROP_COLS = LEAKAGE_COLS + ID_COLS + TEXT_COLS + [
    "patient_name_anon",
    "patient_state",
    "attempted_at",
    "scheduled_at",
]

# --- Categorical columns for encoding ---
CATEGORICAL_COLS = [
    "outcome",
    "direction",
    "whisper_status",
    "cycle_status",
    "day_of_week",
]

# --- The 14 health check-in questions (canonical) ---
HEALTH_QUESTIONS = [
    "How have you been feeling overall?",
    "What's your current weight in pounds?",
    "What's your height in feet and inches?",
    "How much weight have you lost this past month in pounds?",
    "Any side effects from your medication this month?",
    "Satisfied with your rate of weight loss?",
    "What's your goal weight in pounds?",
    "Any requests about your dosage?",
    "Have you started any new medications or supplements since last month?",
    "Do you have any new medical conditions since your last check-in?",
    "Any new allergies?",
    "Any surgeries since your last check-in?",
    "Any questions for your doctor?",
    "Has your shipping address changed?",
]
