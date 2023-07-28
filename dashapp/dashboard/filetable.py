import json

import dash
from dash import dcc, dash_table, Input, Output, html, State
import pandas as pd

from .utilities import (split_filter_part, subdirs_table,
                        save_table, send_get_request)
from .layout import html_layout
from settings import root_path, dbapp_host, dbapp_port


def init_callback(dash_app):
    @dash_app.callback(
        Output('table-filtering-be', "data"),
        Input('table-filtering-be', "filter_query"),
        Input('dataframe-store', 'data'))
    def update_table(filter, data):
        filtering_expressions = filter.split(' && ')
        df = pd.read_json(data)
        dff = df
        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                # this is a simplification of the front-end filtering logic,
                # only works with complete fields in standard format
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]

        return dff.to_dict('records')

    @dash_app.callback(
        Output('table-filtering-be', 'selected_rows'),
        [Input('toggle-all', 'value')],
        [State('table-filtering-be', 'derived_virtual_indices')]
    )
    def update_selected_rows(value,
                             derived_virtual_indices):
        if value == ['Toggle all']:
            return list(range(len(derived_virtual_indices)))
        elif value == []:
            return []
        else:
            return []

    @dash_app.callback(
        Output('loading-2', 'children'),
        [Input('submit-button', 'n_clicks')],
        [State('table-filtering-be', 'derived_virtual_data'),
         State('table-filtering-be', 'selected_rows')]
    )
    def index_new_files(n_clicks,
                        derived_virtual_data,
                        selected_rows):
        if n_clicks is not None and n_clicks > 0:
            save_table(derived_virtual_data, selected_rows)
            send_get_request(f"http://{dbapp_host}:{dbapp_port}/indexnew")
        return True

    @dash_app.callback(
        [Output('dataframe-store', 'data'),
         Output('loading-1', 'children')],
        [Input('search-button', 'n_clicks')],
        [State('input-path', 'value')]
    )
    def store_data(n_clicks,
                   path):
        if n_clicks is not None and n_clicks > 0:
            df = subdirs_table(path)
            outjson = json.dumps(df.to_dict('records'))
        else:
            outjson = json.dumps([])
        return outjson, True


def init_table(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/filetable/",
        index_string=html_layout,
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )
    df = subdirs_table(root_path)
    # Create Dash Layout
    dash_app.layout = html.Div([dcc.Store(id='dataframe-store',
                                          data=json.dumps(df.to_dict('records'))),
                                html.Div([dcc.Input(id='input-path',
                                                    type='path',
                                                    value=root_path,
                                                    placeholder='Enter path...',
                                                    style={'width': '500px'}),
                                          html.Button('Search',
                                                      id='search-button'),
                                          dcc.Loading(id="loading-1")],
                                         style={'flexdirection': 'row'}),
                                dcc.Checklist(['Toggle all'], [],
                                              id='toggle-all'),
                                html.Button('Submit selected',
                                            id='submit-button'),
                                dcc.Loading(id="loading-2"),
                                dash_table.DataTable(
                                    id='table-filtering-be',
                                    style_data={
                                        'whiteSpace': 'normal',
                                        'height': 'auto',
                                        'width': 'auto',
                                    },
                                    columns=[
                                        {"name": i, "id": i} for
                                        i in sorted(df.columns)
                                    ],
                                    filter_action='custom',
                                    filter_query='',
                                    row_selectable='multi'
    )])
    init_callback(dash_app)
    return dash_app.server
