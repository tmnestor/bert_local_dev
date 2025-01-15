#!/usr/bin/env python
# coding: utf-8

### Download pretrained LLM from [Index of /reimers/sentence-transformers/v0.2/](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/)

# download_url="https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/average_word_embeddings_glove.6B.300d.zip"
# download_url="https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/all-MiniLM-L6-v2.zip"

# destination_path="all-MiniLM-L6-v2"
# zip_filename="all-MiniLM-L6-v2.zip"

# # Create the destination directory if it does not exist
# mkdir -p "$destination_path"

# # Download the file quietly
# wget -q -O "$destination_path/$zip_filename" "$download_url"

# # Check if the download was successful
# if [ -f "$destination_path/$zip_filename" ]; then
#     # Unzip the file quietly
#     unzip -q "$destination_path/$zip_filename" -d "$destination_path"
    
#     # Check if the unzip was successful
#     if [ $? -eq 0 ]; then
#         echo "Download and extraction successful."
#     else
#         echo "Extraction failed."
#     fi
    
#     # Remove the zip file after extraction
#     rm "$destination_path/$zip_filename"
# else
#     echo "Download failed."
# fi

import os
import subprocess

# Define variables
download_url = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/all-MiniLM-L6-v2.zip"
destination_path = "/home/jovyan/work/all-MiniLM-L6-v2"
zip_filename = "all-MiniLM-L6-v2.zip"
zip_filepath = os.path.join(destination_path, zip_filename)

# Create the destination directory if it does not exist
os.makedirs(destination_path, exist_ok=True)

# Download the file quietly
download_command = f"wget -q -O {zip_filepath} {download_url}"
download_result = subprocess.run(download_command, shell=True)

# Check if the download was successful
if os.path.isfile(zip_filepath):
    # Unzip the file quietly
    unzip_command = f"unzip -q {zip_filepath} -d {destination_path}"
    unzip_result = subprocess.run(unzip_command, shell=True)
    
    # Check if the unzip was successful
    if unzip_result.returncode == 0:
        print("Download and extraction successful.")
    else:
        print("Extraction failed.")
    
    # Remove the zip file after extraction
    os.remove(zip_filepath)
else:
    print("Download failed.")


# import os
# default is in ~/.cache
# os.environ['HF_HOME'] = '/home/jovyan/cache/'
# os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# ### reload from cache snapshot (need to locate the config.json location)

# MODEL='thenlper/gte-small'
# MODEL='/home/jovyan/cache/hub/models--thenlper--gte-small/snapshots/17e1f347d17fe144873b1201da91788898c639cd'
# gte_model = SentenceTransformer(MODEL)


from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader


# MODEL='thenlper/gte-small'
MODEL = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL)
model

# model = SentenceTransformer('all-MiniLM-L6-v2')
# model


# MODEL = 'all-mpneimport os
# os.environ['HF_HOME'] = '/blabla/cache/'t-base-v2'
# MODEL = r'../all-MiniLM-L6-v2'



################# LOAD SENTENCE TRANSFORMER MODEL ###########################
# Load the embedding model and tokenizer manually

word_embedding_model = models.Transformer(MODEL)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# Assemble the sentence transformer model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

MAX_SEQ_LENGTH = 25

model[0].max_seq_length = MAX_SEQ_LENGTH
model[0].tokenizer.max_seq_length = MAX_SEQ_LENGTH
model[0].do_lower_case = True
model[1].pooling_mode_cls_token=True
model[1].pooling_mode_mean_tokens=False
model[1].pooling_mode_max_tokens=False
model[1].pooling_mode_mean_sqrt_len_tokens=False
model[1].pooling_mode_weightedmean_tokens=False
model[1].pooling_mode_lasttoken=False

print(f"{model=}")

sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('../all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
# Save the model locally
model.save(r'../gte-small')


# ### Using Transformers Library

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

sentences = ["This is an example sentence", "Each sentence is converted"]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


tokenizer = AutoTokenizer.from_pretrained(r'../all-MiniLM-L6-v2')
model = AutoModel.from_pretrained(r'../all-MiniLM-L6-v2')

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

print(f"{encoded_input=}")
with torch.no_grad():
    model_output = model(**encoded_input)
    print(f"{model_output['last_hidden_state'].shape=}")
    print(f"{model_output['pooler_output'].shape=}")
# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
print(f"{sentence_embeddings.shape=}")
print("Sentence embeddings:")
print(sentence_embeddings)



