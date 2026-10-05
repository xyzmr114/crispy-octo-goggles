import polars as pl
import json

df = pl.read_parquet("data/raw/lmsys_chatbot_arena.parquet")
print(f"Total rows: {len(df)}")

# Filter English conversations and non-toxic
df_en = df.filter(pl.col("language") == "English")
print(f"English rows: {len(df_en)}")

# Model counts across model_a and model_b
models_a = df_en["model_a"].to_list()
models_b = df_en["model_b"].to_list()
all_models = models_a + models_b
from collections import Counter
counts = Counter(all_models)
print("\nTop 20 most frequent models in LMSYS Chatbot Arena:")
for m, c in counts.most_common(20):
    print(f"  {m:25s}: {c} battles")

# Check a sample conversation structure
sample_conv = df_en["conversation_a"][0]
print("\nSample conversation_a structure:")
print("Type:", type(sample_conv))
if isinstance(sample_conv, list):
    for turn in sample_conv[:2]:
        print("  role:", turn.get("role"), "| content preview:", str(turn.get("content"))[:80])
