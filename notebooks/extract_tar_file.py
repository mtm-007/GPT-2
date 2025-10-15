import os
import lzma
#from numpy import character
from tqdm import tqdm

def xz_files_in_dir(directory):
    """ to extract .xz file 
    openwebtext dataset is in tar.xz format in 
    link "https://zenodo.org/records/3834942/files/openwebtext.tar.xz?download=1" size:12.9GB """
    files = []
    print("Looking in:", os.path.abspath(directory))
    assert os.path.exists(directory), f"Directory not found: {directory}"

    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory,filename)):
            files.append(filename)
    return files


folder_path = "./openwebtext/"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

files = xz_files_in_dir(folder_path)
total_files = len(files)


split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path,"rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)


with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path,"rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')