import json
from src.models.baseline import BaselineSuite
from sklearn.model_selection import train_test_split

with open("data/processed/benchmark/train.jsonl", "r", encoding="utf-8") as f:
    all_train = [json.loads(l) for l in f]
with open("data/processed/benchmark/test.jsonl", "r", encoding="utf-8") as f:
    all_test = [json.loads(l) for l in f]

all_data = all_train + all_test

qa_data = [r for r in all_data if r["domain"] == "General_QA"]
coding_data = [r for r in all_data if r["domain"] == "Technical_STEM"]
creative_data = [r for r in all_data if r["domain"] in ("Creative_Writing", "Humanities_Reasoning")]

print(f"General QA: {len(qa_data)}, Coding/Technical: {len(coding_data)}, Creative: {len(creative_data)}")

# In-Domain Split for General QA
qa_train, qa_test = train_test_split(qa_data, test_size=0.2, random_state=42, stratify=[r["family"] for r in qa_data])

model = BaselineSuite(max_features=12000)
model.fit([r["llm_output"] for r in qa_train], [r["family"] for r in qa_train])

ev_in = model.evaluate([r["llm_output"] for r in qa_test], [r["family"] for r in qa_test])
ev_coding = model.evaluate([r["llm_output"] for r in coding_data], [r["family"] for r in coding_data])
ev_creative = model.evaluate([r["llm_output"] for r in creative_data], [r["family"] for r in creative_data])

acc_in = ev_in["logistic_regression"]["accuracy"]
f1_in = ev_in["logistic_regression"]["macro_f1"]

acc_code = ev_coding["logistic_regression"]["accuracy"]
f1_code = ev_coding["logistic_regression"]["macro_f1"]

acc_creat = ev_creative["logistic_regression"]["accuracy"]
f1_creat = ev_creative["logistic_regression"]["macro_f1"]

print(f"In-Domain (General QA Test):  Acc = {acc_in:.4f} | Macro-F1 = {f1_in:.4f}")
print(f"OOD Transfer -> Coding/Tech:   Acc = {acc_code:.4f} | Macro-F1 = {f1_code:.4f} (Drop: {(acc_in - acc_code)*100:.1f}%)")
print(f"OOD Transfer -> Creative:      Acc = {acc_creat:.4f} | Macro-F1 = {f1_creat:.4f} (Drop: {(acc_in - acc_creat)*100:.1f}%)")
