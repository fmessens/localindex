import requests
import os
from datetime import datetime

import pandas as pd
import sqlite3

from settings import dbapp_host, dbapp_port, file_ext, indexing_db


def send_get_request(url):
    """Send request to url with try except block.

    Args:
        url (str): url

    Returns:
        (None, dict): response data if 
            request is successful, else None
    """
    try:
        response = requests.get(url)
        response_data = response.json()
        return response_data
    except requests.RequestException as e:
        print(f"Error occurred: {e}")
        return None


def send_query_request(query):
    """Send query request to dbapp.

    Args:
        query (str): query to encode

    Returns:
        dict: response data
    """
    base_url = f"http://{dbapp_host}:{dbapp_port}/queryprocessed"
    encoded_q = requests.utils.quote(query)
    url = f"{base_url}?q={encoded_q}"
    return send_get_request(url)


def encode_pdf_request(pdf_path):
    """Get encoded pdf path to send in request.

    Args:
        pdf_path (str): pdf path

    Returns:
        str: query string
    """
    pdf_path = pdf_path.replace('\\', '/')
    base_url = f"http://{dbapp_host}:{dbapp_port}/showPDFs"
    encoded_path = requests.utils.quote(pdf_path)
    return f"{base_url}?path={encoded_path}"


def subdirs_table(path):
    """Returns a pandas dataframe with the filetree of the path.

    Args:
        path (str): path to directory

    Returns:
        pd.DataFrame: filetree in a dataframe
    """
    df = pd.DataFrame(columns=['name', 'path', 'type'])
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.split('.')[-1] in file_ext:
                df = pd.concat([df,
                                pd.DataFrame({'name': [name],
                                              'path': [os.path.join(root, name)],
                                              'type': [name.split('.')[-1]]})],
                               ignore_index=True)
    return df


def save_table(lsdict, idx):
    """Saves the table to a SQL db.

    Args:
        lsdict (List[Dict]): list of dictionaries representing a dataframe
        idx (int): the index to save
    """
    df = pd.DataFrame(lsdict).iloc[idx]
    conn = sqlite3.connect(indexing_db)
    now = datetime.now()
    name = f"data_{now.strftime('%Y%m%d_%H%M%S')}"
    df.to_sql(name, con=conn, index=False)


operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    """Split filter part into name, operator and value.

    Args:
        filter_part (str): string with filter expression

    Returns:
        Tuple[str]: list with name, operator and value
    """
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return tuple([None] * 3)
