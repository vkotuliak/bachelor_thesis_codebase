import pandas as pd
import matplotlib.pyplot as plt

# file_path = "data/full_data/equipment_database.csv"
# df = pd.read_csv(file_path)

file_path = "data/query_generation/i500_documents.jsonl"
df = pd.read_json(file_path, lines=True)

total_rows = len(df)
missing_counts = df.isnull().sum()
present_counts = total_rows - missing_counts

present_pct = (present_counts / total_rows) * 100
missing_pct = (missing_counts / total_rows) * 100

stats_df = pd.DataFrame(
    {"Present (%)": present_pct, "Missing (%)": missing_pct}
).sort_values(by="Present (%)", ascending=False)

plt.close("all")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot()
stats_df.plot(kind="barh", stacked=True, color=["#2ecc71", "#e74c3c"], ax=ax)

plt.title("Data Completeness per Column", fontsize=15)
plt.xlabel("Percentage of Records (%)")
plt.ylabel("Category (Column Name)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print((df.count() / len(df) * 100).round(1).sort_values(ascending=False))
print("Visualization saved as 'completeness_report.png'")
