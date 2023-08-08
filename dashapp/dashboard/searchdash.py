import os

import dash
from dash import Input, Output, State, html, dcc
import pandas as pd
import openai

from .utilities import send_query_request, encode_pdf_request
from .layout import html_layout

openai.api_key = os.environ.get('OPENAI_API_KEY', 'no-key')


def init_callback(dash_app):
    """initialize callbacks for the dash app

    Args:
        dash_app (dash.Dash): dash app

    Returns:
        None: state functions in the dash app
    """
    @dash_app.callback(
        [Output('full-texts', 'children'),
         Output('pdf-viz', 'children')],
        [Input('submit-button', 'n_clicks')],
        [State('input-text', 'value')]
    )
    def update_output(n_clicks, input_text):
        if n_clicks > 0 and input_text:
            # Here, you should send the input_text to your API and get the response.
            # For demonstration purposes, we assume you get 'response_texts' and 'pdf_locations'
            outdict = send_query_request(input_text)
            assert outdict is not None, "No response received from the API."
            pagefiles = pd.DataFrame(outdict['pagefiles'])
            response_texts = [x for x in outdict['texts']]
            # Create a list of HTML elements to display the response texts
            response_elements = [html.Div([html.P(text),
                                           html.Hr(),
                                           html.Hr()])
                                 for text in response_texts]
            # Update the PDF tabs dynamically based on the number of PDFs received
            pdffiles = pagefiles[pagefiles.file.str.endswith('.pdf')]
            pdf_tabs = [dcc.Tab(label=f'PDF {i+1}',
                                value=f'tab-{i+1}',
                                children=[html.Div([html.P(r['file']),
                                                    html.Iframe(src=(encode_pdf_request(r['file'])
                                                                     + '#page={}'.format(r['page'])),
                                                                style={'width': '100%',
                                                                       'height': '500px'})])])
                        for i, r in pdffiles.iterrows()]

            return response_elements, pdf_tabs

        return (html.P("Please enter a text and click the Submit button."),
                [])

    @dash_app.callback(
        Output('GPT-answers', 'children'),
        Input('full-texts', 'children'),
        State('input-text', 'value')
    )
    def update_GPT_answers(full_texts, query):
        if full_texts is not None and query is not None:
            if openai.api_key == 'no-key':
                return html.P("Please set the OPENAI_API_KEY environment variable.")
            else:
                texts_extyr = [x['props']['children'][0]['props']['children']
                               for x in full_texts]
                content = ".\n\n".join(texts_extyr)+"\n\n"+query+'?'
                print(len(content.split()))
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Use as much the content of my first message as possible"},
                            {"role": "user", "content": content}
                        ])
                    return html.P(completion.choices[0].content)
                except openai.error.RateLimitError as e:
                    return html.P(str(e)+"\n Alternatively, you can set a lower value of samples_return in settings.py.")
        else:
            return html.P("Please enter a text and click the Submit button.")


def init_search(server):
    """Create a Plotly Dash dashboard.

    Args:
        server (Flask): Flask server

    Returns:
        Dash.dash.server: Dash application server
    """
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/searchapp/",
        index_string=html_layout,
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )
    # Create Dash Layout
    dash_app.layout = html.Div([html.Div([
        dcc.Input(id='input-text',
                  type='text',
                  placeholder='Enter text...',
                  style={'width': '500px', 'height': '100px'}),
        html.Button('Submit',
                    id='submit-button',
                    n_clicks=0),
        html.Div(id='output-div',
                 children=[html.H3('Output text:'),
                           dcc.Tabs(id="tabs-output", value='full-texts', children=[
                               dcc.Tab(label='Full texts', id='full-texts',
                                       style={'width': '200px'}),
                               dcc.Tab(label='GPT answers', id='GPT-answers',
                                       style={'width': '200px'}),
                           ],
                     style={'width': '100%'})
                 ])
    ],
        style={'display': 'inline-block',
               'verticalAlign': 'top',
               'marginRight': '20px',
               'width': '40%'}),
        html.Div(id='pdf-viz',
                 style={'display': 'inline-block',
                        'verticalAlign': 'top',
                        'marginRight': '20px',
                        'width': '100%'})],
        style={'display': 'flex',
               'flexDirection': 'row'})
    init_callback(dash_app)
    return dash_app.server
