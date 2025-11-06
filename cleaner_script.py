import pandas as pd
import langid
import nltk
from pathlib import Path
from tqdm import tqdm

script_dir = Path(__file__).parent
data_path = script_dir / "data" / "reviews_hotel1.csv"

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# === Load dataset ===
df = pd.read_csv(data_path)
print("Read the dataset just now")

# === Drop rows with missing reviews ===
df = df[df['reviews.text'].notna()]

# === Keep only rows with ratings 1-5 ===
df = df[df['reviews.rating'].isin([1.0, 2.0, 3.0, 4.0, 5.0])]

# === Optional: subset for faster experimentation ===
# df = df.head(5000)
print("before keeping only english", df.shape)
tqdm.pandas(desc="Detecting English reviews")
df = df.head(5000).reset_index(drop=True)
print("before keeping only english", df.shape)

# === Keep only English reviews ===
df = df[df['reviews.text'].progress_apply(lambda x: langid.classify(str(x))[0] == 'en')]
print("before combine")
# === Combine title + review text into one column ===
df['Review'] = df['reviews.title'].fillna('') + '\n' + df['reviews.text'].fillna('')
df.drop(columns=['reviews.title', 'reviews.text'], inplace=True)

# === Rename rating column for convenience ===
df.rename(columns={'reviews.rating': 'Rating'}, inplace=True)

# === Reset index ===
df = df.reset_index(drop=True)

# === Final cleaned dataset ===
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print(df.head())

# === Ready for CV ===
# X = df['Review']  (input features)
# y = df['Rating']  (target labels)
# Save cleaned dataframe
df.to_csv(script_dir / 'data' / "reviews_hotel1_clean.csv", index=False)

print("Cleaned dataset saved as reviews_hotel1_clean.csv")