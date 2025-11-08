import json

input_path = "./data/corpus/books_preprocessed.jsonl"
output_path = "./data/corpus/books_for_solr.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        doc = json.loads(line)
        fout.write(json.dumps({"add": {"doc": doc}}) + "\n")
