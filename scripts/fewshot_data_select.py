import json
import random

random.seed(42)


json_file = "/yinghepool/downstream_data/data_path/气胸积液/data_2d_气胸积液_train.json"

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    data_list = data["data_list"]

sample_size = max(1, int(len(data_list) * 0.1))
sampled_data = random.sample(data_list, sample_size)

new_json_data = {
    "csv_file_path": data["csv_file_path"],
    "metainfo": data["metainfo"],
    "data_list": sampled_data
}
    
output_path = "./splits/qixiongjiye_10%_pos.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_json_data, f, ensure_ascii=False, indent=4)

print(f"已抽取10%的样本（{sample_size}个）并保存到 {output_path}")