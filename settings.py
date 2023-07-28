import os

file_ext = ['pdf']

root_path = os.path.expanduser('~')+'/Documents'

indexing_db = 'data/indexing.db'

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
embedding_location = 'data/embeddings_dataset.json'
model_summ = "mrm8488/t5-base-finetuned-summarize-news"

window = 384
step = 128
samples_return = 5

dashapp_port = 5000
dbapp_port = 5002

dashapp_host = "0.0.0.0"
dbapp_host = "127.0.0.1"
