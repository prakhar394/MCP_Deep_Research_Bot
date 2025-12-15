# scripts/build_gt_addhealth.py
import json
import pandas as pd
from pathlib import Path

INPUT_XLSX = Path("/Users/devdattgolwala/Documents/MCP_Deep_Research_Bot-multimode-bot/data/Ground Truth (40 Subsample).xlsx")  # adjust path if needed
OUTPUT_JSONL = Path("/Users/devdattgolwala/Documents/MCP_Deep_Research_Bot-multimode-bot/data/gt_addhealth_qa.jsonl")

QUESTION_FIELDS = {
    "What area of research is this paper investigating? What hypothesis is this paper testing?": "area_hypothesis",
    "Does this paper use a single wave or multiple waves of Add Health data?": "waves",
    "What population is this paper generalizable to?": "population",
    "What results did the researchers find based on their analysis?": "results",
    "What analysis method was used by the researchers?": "analysis_method",
    "What limitations does this paper have?": "limitations",
}

def main():
    df = pd.read_excel(INPUT_XLSX)

    examples = []
    for row_idx, row in df.iterrows():
        title = str(row["Article TItle"]).strip()
        doi = str(row["DOI"]).strip()

        for col_name, q_type in QUESTION_FIELDS.items():
            gold = row.get(col_name)
            if not isinstance(gold, str) or not gold.strip():
                continue

            # You can keep the column text as the question, or slightly rephrase:
            query = f"{col_name} (Paper title: {title})"

            ex = {
                "id": f"addhealth_{row_idx+1}_{q_type}",
                "dataset": "addhealth",
                "query": query,
                "gold_answer": gold.strip(),
                "meta": {
                    "paper_index": int(row_idx + 1),
                    "paper_title": title,
                    "doi": doi,
                    "question_type": q_type,
                },
            }
            examples.append(ex)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} GT examples to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()