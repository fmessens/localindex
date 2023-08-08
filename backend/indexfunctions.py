import os
import json

import pandas as pd
import numpy as np
import sqlite3
import PyPDF2
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead

from settings import (indexing_db, model_ckpt, embedding_location,
                      window, step, samples_return, model_summ)


def create_flecontext(file_path):
    """Convert file path to add to text string"""
    flecontxt = (file_path.replace('C:/', '')
                 .replace('/', ' ').replace('\\', ' '))
    return flecontxt


def pdf_lines_text(file_path, window, step):
    """convert pdf text to dataframe with text chunks
    for embedding

    Args:
        file_path (str): path to pdf file
        window (int): window size of text chunks
        step (int): step size of text chunks

    Returns:
        pd.DataFrame: dataframe with text chunks, 
            start and end line, page and token index
    """
    # put all the text lines of a pdf file in dataframe
    fullseq = []
    linetotal = 0
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text = page.extract_text()+'\n'
            lineseq = text.split('\n')
            lineseq2 = [x+'§§' for x in lineseq]
            tokenseq = [(linetotal+i,
                         y.replace('§§', '\n'))
                        for i, x in
                        enumerate(lineseq2)
                        for y in
                        x.split()]
            tokendf = pd.DataFrame(tokenseq,
                                   columns=['line',
                                            'token'])
            linetotal = tokendf.line.max()+1
            tokendf['page'] = page_number
            fullseq.append(tokendf)
    fulldf = pd.concat(fullseq).reset_index(drop=True)
    # get the tokens, pages, lines and index into a list
    tokenseq = fulldf.token.tolist()
    pageseq = fulldf.page.tolist()
    lineseq = fulldf.line.tolist()
    idx = fulldf.index.tolist()
    # add the file path to the text in order to give extra context
    flecontxt = create_flecontext(file_path)
    # adapt the window size to account for the flecontxt
    window = window-len(flecontxt.split())
    # total length of the text
    seqlen = len(tokenseq)
    # get chunks of text based on window and step size
    pdfchunks = [' '.join(tokenseq[i:i+window])
                 for i in range(0, seqlen, step)]
    # add flecontxt to the chunks
    pdfchunks2 = [flecontxt + ':\n ' + x
                  for x in pdfchunks]
    # get the start and end lines, pages and index of the chunks
    startpages = [pageseq[i] for i in
                  range(0, seqlen, step)]
    startlines = [lineseq[i] for i in
                  range(0, seqlen, step)]
    starrtidx = [idx[i] for i in
                 range(0, seqlen, step)]
    endpages = [pageseq[-1] if i+window > seqlen
                else pageseq[i+window]
                for i in range(0, seqlen, step)]
    endlines = [lineseq[-1] if i+window > seqlen
                else lineseq[i+window] for i in
                range(0, seqlen, step)]
    endidx = [idx[-1] if i+window > seqlen
              else idx[i+window] for i in
              range(0, seqlen, step)]
    # put all this info in a dataframe
    pdfchunksdf = pd.DataFrame({'text': pdfchunks2,
                                'startlines': startlines,
                                'startpages': startpages,
                                'startidx': starrtidx,
                                'endlines': endlines,
                                'endpages': endpages,
                                'endidx': endidx})
    # return the dataframe
    return pdfchunksdf


def index_file(fle, window, step):
    """index a file to embedding database

    Args:
        fle (str): file path
        window (int): window size of text chunks
        step (int): step size of text chunks

    Raises:
        ValueError: If file type not supported

    Returns:
        Dataset: Embeddings dataset (huggingface datasets)
    """
    ftype = fle.split('.')[-1]
    if ftype == 'pdf':
        chunks = pdf_lines_text(fle, window, step)
        chunks['file'] = fle
    else:
        raise ValueError(f'File type {ftype} not supported')
    chunks_dataset = Dataset.from_pandas(chunks)
    embedder = Embedder(model_ckpt)
    embeddings_dataset = chunks_dataset.map(
        lambda x: {"embeddings": embedder.get_embeddings(x["text"])
                   .detach()
                   .numpy()[0]})
    return embeddings_dataset


class Embedder():
    """Class for embedding text using a transformer model
    """
    def __init__(self, model_ckpt):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)

    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output)


def index_new_files():
    """index new files in the indexing database

    Returns:
        dict: progress of indexing
    """
    # connect to sqlite db
    conn = sqlite3.connect(indexing_db)
    # get all the tables with 'data_' in the name
    all_submits = pd.read_sql("SELECT name FROM sqlite_master", con=conn)
    all_submitss = all_submits[all_submits['name'].str.contains('data_')]
    # take the last version of the table
    all_submitss = all_submitss.sort_values('name').reset_index(drop=True)
    last_submit = all_submitss['name'].iloc[-1]
    # read this table into a dataframe
    filesdf = pd.read_sql(f"SELECT * FROM {last_submit}",
                          con=conn)
    # if there is not yet a table called 'files_embedded' create it
    if 'files_embedded' not in all_submits['name'].tolist():
        (pd.DataFrame({'file': []})
         .to_sql('files_embedded', con=conn,
                 if_exists='replace'))
    # read the table 'files_embedded' into a dataframe
    filesemb = pd.read_sql(f"SELECT * FROM files_embedded",
                           con=conn)
    # filter dataframe to only include files that are not yet embedded
    filesdfsel = filesdf[~filesdf['path'].isin(filesemb['file'])]
    # loop over the files and embed them
    for file in filesdfsel.path:
        print(file)
        filesprogress = {}
        try:
            embds = index_file(file, window, step)
            embdsdf = embds.to_pandas()
            with open(embedding_location, 'a') as f:
                embdsdf.to_json(f, orient='records', lines=True)
            (pd.DataFrame({'file': [file]})
             .to_sql('files_embedded', con=conn,
                     if_exists='append'))
            filesprogress[file] = 'success'
        except Exception as e:
            filesprogress[file] = e
    if len(filesdfsel) == 0:
        filesprogress = {'no new files': 'success'}
    return filesprogress


class QueryIndex:
    """a class for querying the embedding dataset
    """
    def __init__(self):
        if not os.path.exists(embedding_location):
            usefile = 'data/placeholder.json'
        else:
            usefile = embedding_location
        self.embeddings_dataset = Dataset.from_json(usefile)
        self.embeddings_dataset.add_faiss_index(column="embeddings")
        self.embedder = Embedder(model_ckpt)

    def query(self, query):
        """perform query on embedding dataset

        Args:
            query (str): the query

        Returns:
            Tuple[List[str], List[float]]: text samples and score
        """
        query_embedding = (self.embedder
                           .get_embeddings([query])
                           .detach().numpy())
        scores, samples = (self.embeddings_dataset
                           .get_nearest_examples(
                               "embeddings",
                               query_embedding,
                               k=samples_return
                           ))
        return samples, scores

    @staticmethod
    def cluster_texts(samples):
        """Cluster text samples

        Args:
            samples (List[dict]): List of dicts with keys 'file', 'startidx'

        Returns:
            List[dict]: List of dicts with keys 'file', 'startidx', 'endidx'
        """
        samples = pd.DataFrame(samples)
        samples = (samples.sort_values(['file', 'startidx'])
                   .reset_index(drop=True))
        texts = []
        for g, df in samples.groupby('file'):
            endix = 0
            for i, x in df.iterrows():
                if x['startidx'] >= endix:
                    texts.append(x['text'])
                    endix = x['endidx']
                else:
                    flecntx = create_flecontext(g)
                    flecntxf = flecntx + ':\n '
                    texts[-1] = texts[-1] + ' ' + x['text'][len(flecntxf):]
                    endix = x['endidx']
        return texts

    @staticmethod
    def cluster_pages(samples):
        """Cluster pages

        Args:
            samples (List[dict]): List of dicts with keys 'file', 'startpages'

        Returns:
            pd.DataFrame: dataframe with columns 'page', 'file'
        """
        samples = pd.DataFrame(samples)
        pages = samples['startpages'].tolist()
        files = samples['file'].tolist()
        pagefiles = (pd.DataFrame({'page': pages,
                                   'file': files}
                                  )
                     .drop_duplicates()
                     .sort_values(['file', 'page'])
                     .reset_index(drop=True))
        return pagefiles

    def processed_query(self, query):
        """Perform a query and process the results

        Args:
            query (str): query

        Returns:
            Tuple[List[dict]]: texts, pagefiles, scores
        """
        samples, scores = self.query(query)
        texts = self.cluster_texts(samples)
        pagefiles = self.cluster_pages(samples)
        return texts, pagefiles, scores


class Summarizer:
    """a class for summarizing text with huggingface transformers
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_summ)
        self.model = AutoModelWithLMHead.from_pretrained(model_summ)

    def summarize(self, text, max_length=150):
        input_ids = self.tokenizer.encode(text, return_tensors="pt",
                                          add_special_tokens=True)
        generated_ids = self.model.generate(input_ids=input_ids,
                                            num_beams=2,
                                            max_length=max_length,
                                            repetition_penalty=2.5,
                                            length_penalty=1.0,
                                            early_stopping=True)
        preds = [self.tokenizer.decode(g, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True)
                 for g in generated_ids]
        return preds[0]
