from pyclassify import kNN
from pyclassify.utils import read_config,  read_file
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Configuration file, relative to the project root and without extension")
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(script_dir, "..", args.config))
print(config_path)
config_dict = read_config(config_path)

kNNclassifier = kNN(config_dict['k'])
data, labels = read_file(os.path.abspath(os.path.join(script_dir, "..", config_dict["dataset"])))
train_fraction = 0.8
train_amount = int(train_fraction*len(data))
train_data, test_data = data[:train_amount], data[train_amount:]
train_labels, test_labels = labels[:train_amount], labels[train_amount:]
predicted_labels = kNNclassifier((train_data, train_labels), test_data)

accuracy = 0 
for i in range(len(predicted_labels)):
    accuracy+=(predicted_labels[i]==test_labels[i])

accuracy/=len(predicted_labels)
print(f"accuracy: {accuracy}")