from src.smear_function import smear_pipeline
import argparse
import yaml
from src.loader import Loader
import os
import sys
import json
import time

def arguments_parser():
    """
    Parse arguments from config file
    """

    parser = argparse.ArgumentParser('Tuberculosis Detection')
    parser.add_argument('config', type=str, default='configs/thresholding.yaml',
                        help='configure file for thresholding experiments')
    return parser

parser = arguments_parser()
pars_arg = parser.parse_args()


with open(pars_arg.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

files = os.listdir('/mnt/storage/TBProject/TB_sample/2022-06-29')

print("The files in the directory are as follows:", files)

# create a smears.json file with all the smears avaiable in the folder and their severity

smear_to_severity = {}
for i in range(5):
    smear_to_severity[str(i)] = []

data = {}

for i, word in enumerate(files):
    severity = word.split('_')[-2]
    data[word] = {}
   
    
    if severity == "MTB" or  (" " in severity) or int(severity) > 5:
        print("This smear doesn't have a severity rating")
        print("The smear is:", word)
        data[word]['severity'] = "None"
        continue
    data[word]['name'] = word
    data[word]['severity'] = severity
    
    data[word]['path'] = "/mnt/storage/TBProject/TB_sample/2022-06-29/" + word
    smear_to_severity[severity].append(word)


with open("smears.json", "w") as f:
    json.dump(data, f)
print("Smears.json file created")

if os.path.exists("smears_json") == False:
    os.mkdir("smears_json")

for i, name in enumerate(data.keys()):
    if os.path.exists("smears_json/" + name + ".json") == True:
        print("Smear", name, "already processed\n")
        continue
    print("Starting to count bacilli in smear", i+1, "of", len(data.keys())+1 , "...")
    path = data[name]["path"]
    ld = Loader(path, 'None')
    ld.load()
    img = ld.data_array
    start_time = time.time()
    num_bacilli, tiles_bacilli, total_objects = smear_pipeline(config, img, ld)
    end_time = time.time()
    data[name]["tot_num_of_bacilli"] = num_bacilli
    data[name]["tot_num_of_objects"] = total_objects
    data[name]["time_to_process_in_seconds"] = end_time - start_time
    data[name]["time_to_process_in_minutes"] = (end_time - start_time) / 60
    data[name]["bacilli_per_single_tile"] = tiles_bacilli
    
    with open("smears_json/" + name + ".json", "w") as f:
        json.dump(data[name], f)
    print("Smear", name, "processed\n")
