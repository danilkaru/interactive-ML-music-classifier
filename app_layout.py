# This file contains the Dash app layout, which represents all of the interactive
# and visual components of the user interface which are displayed to the user at
# http://127.0.0.1:8050/ (default Dash URL). 


import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context
import dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from config import config,ui_layout_config

layout = html.Div([
    # Title
    html.H1("Interactive ML Music Classifier", style={'textAlign': 'center'}),


    html.Div([
        # Left: Upload Button
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload MAEST Embedding JSON File'),
                multiple=False
            )
        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),


        # Right: Dropdown + Download Button
        html.Div([
            dcc.RadioItems(
                id='json-log-radio-items',
                options=[
                    {'label': 'Verbose', 'value': 'Verbose'},
                    {'label': 'Stats', 'value': 'Stats'}
                ],
                value='Verbose',
                labelStyle={'display': 'inline-block', 'marginRight': '15px'}
            ),
            html.Button('Download JSON logs', id='download-json-button', n_clicks=0),
            dcc.Download(id="download-json"),
        ], style={'float': 'right', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'marginBottom': '30px', 'overflow': 'hidden'}),

    # Hidden Stores
    dcc.Store(id='selected_points', storage_type='session'),
    dcc.Store(id='training-state', storage_type='memory'),
    dcc.Store(id='hover-audio-path', storage_type='memory'),
    dcc.Store(id='store-annotated'),
    dcc.Store(id='store-predicted'),
    html.Div(id='dummy-output', style={'display': 'none'}),

    # Problematic tracks statistics section
    html.Div([
        html.H3("Problematic Tracks Statistics", style={'marginBottom': '10px'}),
        html.Div(id='problematic-tracks-stats', children="Statistics will appear after training the classifier"),
    ], style={'marginBottom': '10px', 'padding': '10px', 'border': '2px ridge blue', 'borderRadius': '5px'}),
    
    html.Div([
        # Class name; input field and assign button
        html.Div([
            dcc.Input(id='class-name', type='text', placeholder='Enter Class Name', style={'marginRight': '10px'}),
            html.Button('Assign Class', id='assign-class', n_clicks=0),

        ], style={'marginTop':'15px','marginBottom': '0px'}),

        # Text input in which user can provide number of tracks highlighted by CPU/DC heuristics
        html.Div([
            html.Label("Type in number of tracks to be marked as problematic (per heuristic): "),
            dcc.Input(id='input_num_problematic', type='number', value=config["default_problematic_tracks_num"], min=0, step=1),                     
            html.Div(id='out-num-problematic'),
        ],style={'marginTop':'15px','marginLeft': '5px', 'marginBottom': '0px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'}),
    
    # IMPORTANT: 2D plot for displaying music collection embedding space
    # NOTE: lasso2d and select2d added to support data point selection
    dcc.Graph(id='umap-plot', config={'modeBarButtonsToAdd': ['lasso2d', 'select2d']}, clear_on_unhover=True),

    # Components for handling audio playback
    html.Div([
        html.Div([
            html.Label("Play audio on hover?"),
            dcc.RadioItems(
                id='audio-play-select',
                options=[
                    {'label': 'Yes', 'value': 'Yes'},
                    {'label': 'No', 'value': 'No'}
                ],
                value='Yes No',
                labelStyle={'display': 'inline-block', 'marginRight': '15px'}
            ),
        ], style={'display': 'inline-block'}),
        html.Div([
            dcc.Input(id='audio-path', type='text', placeholder='Enter Audio Path', style={'marginLeft': '10px', 'marginRight': '100px', 'width':'300px'}),            
        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),        

    ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Br(), html.Br(),
    

    html.Div([
        html.Div([
            html.Button('Reset Plot', id='reset-button',style={'marginRight': '10px'}),
            html.Button('Train & Predict', id='train-mlp', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('Reject Predictions', id='reject-predictions', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('Accept Predictions', id='accept-classifications', n_clicks=0),
        ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
    
        html.Div([
            html.Button('Download results', id='download-button', n_clicks=0, disabled=False),
            dcc.Download(id="download-annotated"),
        ], style={'float': 'right', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'marginBottom': '30px', 'overflow': 'hidden'}),

    # UI datatables
    html.H1("Data Table (Annotated Songs)"),
    html.Div(id='debug-output', style={'marginBottom': '10px', 'color': 'red'}),

    dash_table.DataTable(
        id='selected-data-table',
        columns=[
            {"name": "ID", "id": "ID"},
            {"name": "Index", "id": "Index"},
            {"name": "Class", "id": "Class"},
            {"name": "Audio Path", "id": "AudioPath"},
            {"name": "Audio", "id": "Audio", "presentation": "markdown"},
            {"name": "Track Status", "id": "TrackStatus"},
        ],
        data=[],
        editable=True,
        row_deletable=True,
        markdown_options={"html": True},
        sort_action="native",
        filter_action='native',
        style_table={'height': 250, 'overflowX': 'auto', 'overflowY': 'scroll'},
    ),

    html.Br(),

    html.H1("Data Table (Predicted Songs)"),
    dash_table.DataTable(
        id='predicted-data-table',
        columns=[
            {"name": "ID", "id": "ID"},
            {"name": "Index", "id": "Index"},
            {"name": "Class", "id": "Class"},
            {"name": "Audio Path", "id": "AudioPath"},
            {"name": "Audio", "id": "Audio", "presentation": "markdown"},
            {"name": "Probabilities", "id": "Probabilities"},
            {"name":"Cosine Distance to Nearest Annotated Point", "id":"CosineDistanceToCentroid", "type": "numeric"},
            {"name": "Track Status", "id": "TrackStatus"},
        ],
        data=[],
        editable=True,
        row_deletable=True,
        markdown_options={"html": True},
        sort_action="native",
        filter_action='native',
        style_table={'height': 250, 'overflowX': 'auto', 'overflowY': 'scroll'},
    ),

    html.Div(id='debug-store-output', style={'marginBottom': '10px', 'color': 'red'}),
])