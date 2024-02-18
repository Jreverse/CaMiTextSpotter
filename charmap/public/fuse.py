import os
import json

jsons = ['arabic.json', 'bengali.json', 'chinese.json', 'devanagari.json', \
        'hangul.json', 'japanese.json', 'latin.json', 'symbol.json']
output = 'generic.json'
idx = 1
json_dict = {}
charset_union = set()

for filename in jsons:
    with open(filename,'r', encoding='utf-8') as f:
        data = json.load(f)
        
    json_dict[filename] = data

num_list = {}
for k, v in json_dict.items():
    num_list[k] = len(v.keys())
    charset_union.update(v.keys())

print(len(charset_union))
print(num_list)
print(sum(num_list.values()))

import ipdb;ipdb.set_trace()
