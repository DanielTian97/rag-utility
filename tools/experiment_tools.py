def update_json_result_file(file_name, result_to_write):
    import json
    
    f = open(file_name, "w+", encoding='UTF-8')
    json.dump(result_to_write, f, indent=4)
    f.close()