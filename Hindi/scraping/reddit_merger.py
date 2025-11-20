import json, glob

merged = []
files = glob.glob("/content/drive/MyDrive/hyigiea/reddit/*.jsonl")

id_counter = 1
for file in files:
    with open(file, "r") as f:
        for line in f:
            entry = json.loads(line)
            entry["id"] = id_counter
            id_counter += 1
            merged.append(entry)

with open("/content/drive/MyDrive/hyigiea/data/reddit_merged.jsonl", "w") as f:
    for e in merged:
        json.dump(e, f, ensure_ascii=False)
        f.write("\n")

print("Merged:", len(merged))
