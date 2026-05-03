RANDOM_STATE = 42

DATA_PATH = "cirrhosis.csv"

TARGET_COLUMN = "Status"

SKEWED_COLS = ["Bilirubin", "Cholesterol", "Copper", "Alk_Phos", "SGOT", "Tryglicerides"]

NUMERIC_FEATURES = ["Age_years", "Bilirubin", "Albumin", "Prothrombin", "Platelets", "N_Days"]

NUMERIC_COLS = [
    "Age_years", "N_Days", "Bilirubin", "Cholesterol", "Albumin",
    "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
]

TOP_FEATURES = ["log_Bilirubin", "Age", "Stage", "Prothrombin"]