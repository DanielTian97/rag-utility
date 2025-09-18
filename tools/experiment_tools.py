def update_json_result_file(file_name, result_to_write):
    import json
    
    f = open(file_name, "w+", encoding='UTF-8')
    json.dump(result_to_write, f, indent=4)
    f.close()

def update_jsonl_result_file(file_name, qid, varying_context_result):
    import json
    
    record = {"qid": qid, **varying_context_result}
    with open(file_name, "a", encoding="UTF-8") as f:
        f.write(json.dumps(record) + "\n")

def read_exist_qids_from_jsonl(opened_file):
    import json
    """
    Reads a JSONL file and returns:
      - existed_qids_list: list of all qid values (from "id" field)
      - existed_qids: number of records
    If the file doesn't exist yet, returns empty list and 0.
    """
    
    existed_qids_list = []
    for line in opened_file:
        record = json.loads(line)
        existed_qids_list.append(record["qid"])
    
    return existed_qids_list, len(existed_qids_list)

def json_to_jsonl(json_file: str, jsonl_file: str):
    """
    Convert a dict-style JSON file into JSON Lines format.
    Each key of the dict becomes an "id" field in JSONL.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)   # {qid: varying_context_result}

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for qid, varying_context_result in data.items():
            record = {"qid": qid, **varying_context_result}
            f.write(json.dumps(record) + "\n")