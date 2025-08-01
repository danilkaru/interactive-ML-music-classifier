# This file contains all of the Dash callback functions which have been implemented
# for the user interface. The Dash callback functions are connected to components
# in the UI (see Dash app layout in app_layout.py); whenever the state of one of
# the components in the app layout is changed, the functions which depend on the 
# state of that component are automatically run. 


# ----------------------------------- IMPORTS -----------------------------------
# Dash imports
import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

# Plotly imports
import plotly.express as px

# Necessary Python module imports
import os
import numpy as np
import pandas as pd
import json 
import datetime
import copy
import time
# from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from scipy.spatial.distance import cosine
from flask import send_file

# Local repository imports
import json_helper as jh
import umap_helper as uh
import embedding_helper as eh
import ml_helper as mh
import text_helper as th

from config import config,ui_layout_config
from data import data_store


json_log = {
    "Software Version": config["version"],
    "Username": config["user"],
    "Events": []
}

stats_log = {}



def get_callbacks(app):
    # ----------------------------------- UPDATE PLOT -----------------------------------

    @app.callback(
        Output('umap-plot', 'figure'),
        Output('debug-output', 'children'), 
        Output('store-annotated', 'data'),
        Output('store-predicted', 'data'),
        Input('upload-data', 'contents'),
        Input('reset-button', 'n_clicks')  
    )
    def update_plot(contents, reset_clicks):
        """
        Function for updating 2D plot. Is called after user uploads embeddings, 
        trains MLP classifier, (re-)annotates data points, or resets the plot. 
        """
        # Start process time measurement
        s_time = time.time()

        # Read out which action triggered this function
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

        # Case 1: Handle initial data upload or plot reset
        if (contents and triggered_id == 'upload-data') or triggered_id == 'reset-button':
            # Process the uploaded .json file 
            embeddings, ids, bpm_values = jh.process_json(contents)
            # Error handling
            if embeddings is None:
                return px.scatter(title="Error processing JSON file"), "Error processing JSON file"
            
            # Calculate data point sizes from BPM/embedding variance
            embedding_variances = eh.compute_MAEST_embedding_space_variance(embeddings=embeddings)
            embedding_variances_scaled = uh.scale_values(values = np.array(embedding_variances), 
                                                        offset = ui_layout_config["size_offset"], 
                                                        scale = ui_layout_config["size_scale"]
            )
            bpm_scaled = uh.scale_values(values = np.array(bpm_values), 
                                        offset = ui_layout_config["size_offset"],
                                        scale = ui_layout_config["size_scale"]
            )

            # Compute 2D UMAP representation of embedding space
            embeddings_2d, _ = uh.apply_umap(embeddings)

            # Initialize DataFrame
            df = pd.DataFrame({
                "x": embeddings_2d[:, 0], 
                "y": embeddings_2d[:, 1], 
                "Class": "?", #Initialise class names with generic name "?"
                "ClassDisplay": "?", 
                "ID": ids, #Song IDs
                "BPM_size": bpm_scaled,
                "Embedding_variance_size": embedding_variances_scaled,
                "Probabilities": ["N/A"] * len(ids), 
                "IsProblematic": ["N/A"] * len(ids),
                "Opacity": [0.2] * len(ids)  #TO DO: add as adjustable parameter to config.py
            })

            # Update data store
            data_store["embeddings"] = embeddings
            data_store["embeddings_2d"] = embeddings_2d
            data_store["df"] = df
            data_store["assigned_classes"] = {}
            data_store["colors"] = {}
            data_store["ids"] = ids
            data_store["bpms"] = bpm_values
            data_store["bpms_sizes"] = bpm_scaled
            data_store["embedding_variances"] = embedding_variances
            data_store["embedding_variances_sizes"] = embedding_variances_scaled
            data_store["predicted_classes"] = {}
            data_store["accepted_predictions"] = False
            data_store["data_store_number"] = 0

            # Display amount of processed embeddings in the user interface (debug-output component)
            debug_info = f"Processed {len(embeddings)} embeddings with {len(ids)} IDs"
        
        # Case 2: Any plot update after initial data upload
        elif data_store.get("df") is not None:
            df = uh.update_dataframe(df=data_store["df"],data_store=data_store,include_predictions=True)
            data_store["df"] = df
            debug_info = f"Using existing data with {len(df)} points"
        else:
        # Case 3: No data uploaded yet
            return px.scatter(title="Upload Data to Visualize"), "", None, None
        
        # Predicate data point size on size_selection param in config.py
        size_selection = config["size_selection"]
        size_column = "BPM_size" if size_selection == 'BPM' else "Embedding_variance_size"
        
        # Create figure (2D scatter plot)
        fig = uh.create_figure(df = df, size_column = size_column, accepted = False) 

        # User annotated data
        annotated_data = []
        for i, row in data_store["df"].iterrows():
            if row["Class"] != "?":  # Append all data which has an updated class name
                annotated_data.append({"ID": row["ID"], "Class": row["Class"]})

        # ML predictions for data points not annotated by the user
        predicted_data = []
        predicted_classes = data_store.get("predicted_classes", {})
        for idx, pred_class in predicted_classes.items():
            prob = data_store["df"].iloc[idx]["Probabilities"]
            predicted_data.append({
                "ID": data_store['ids'][idx],
                "PredictedClass": pred_class,
                "Probabilities": prob
            })
            
        return fig, debug_info, annotated_data, predicted_data

    # ----------------------------------- ASSIGN USER-DEFINED CLASSES -----------------------------------

    @app.callback(
        Output('umap-plot', 'figure', allow_duplicate=True),
        Output('selected-data-table', 'data', allow_duplicate=True),
        Input('assign-class', 'n_clicks'),
        State('umap-plot', 'selectedData'),
        State('class-name', 'value'),
        prevent_initial_call=True
    )
    def assign_class(n_clicks, selectedData, class_name):
        """
        Function which handles the assigninment of classes to data points 
        by the user. Requires the selection of data points ('selectedData'),
        text input in class-name component, and click on the 'Assign Class'
        button.
        """
        # Start process time measurement
        s_time = time.time()

        # Read out which action triggered this function
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

        # Error handling; prevent from updating if no data points are selected
        # or no class name has been provided
        if not selectedData or not class_name:
            return dash.no_update, dash.no_update
        
        # Store IDs of selected data points
        selected_ids = [point['customdata'][0] for point in selectedData['points']]
        # print(f"Selected IDs: {selected_ids}") #debug print statement
        
        # # Store indices of selected data points
        selected_indices = [data_store["ids"].index(music_id) for music_id in selected_ids if music_id in data_store["ids"]]
        # print(f"Selected indices: {selected_indices}") #debug print statement

        # Assign colors to new classes
        if class_name not in data_store["colors"]:
            data_store["colors"][class_name] = px.colors.qualitative.Plotly[len(data_store["colors"]) % len(px.colors.qualitative.Plotly)]
        
        # Assign and store class name for selected data points
        for idx in selected_indices:
            data_store["assigned_classes"][idx] = class_name
        
        # Create copy of df
        # NOTE: Necessary for preventing bugs in data updates
        df = data_store["df"].copy()
        
        # Update class information based on updated data_store["assigned_classes"] state
        for idx, cls in data_store["assigned_classes"].items():
            if idx < len(df):
                df.loc[idx, "Class"] = cls
                df.loc[idx, "ClassDisplay"] = cls[:2] if cls else "?" # Set displayed class name
        
        # Store updated copy of df
        data_store["df"] = df
        
        # Update datatables in UI based on new assigned classes
        assigned_classes_copy = dict(data_store["assigned_classes"])
        table_data = uh.create_table_data(predicted_classes=assigned_classes_copy,ids=data_store["ids"],config=config,data_store=data_store,table_type="annotated")

        # Predicate data point size on size_selection param in config.py
        size_selection = config["size_selection"]
        size_column = "BPM_size" if size_selection == 'BPM' else "Embedding_variance_size"

        # Create figure (2D scatter plot)
        fig = uh.create_figure(df = df, size_column = size_column, accepted = False) 
        
        # Stop process time measurement
        e_time = time.time()

    
        event_info = {
            "Timestamp": str(datetime.datetime.now().isoformat()),
            "Event Type": "Assign Class",
            "Event Process Time": str(round(e_time - s_time, 5)),
            "Triggered Dash ID": str(triggered_id)
        }

        json_log["Events"].append(event_info)

        return fig, table_data

    # ----------------------------------- ML TRAIN AND PREDICT -----------------------------------

    @app.callback(
        Output('umap-plot', 'figure', allow_duplicate=True),
        Output('predicted-data-table','data', allow_duplicate=True),
        Output('training-state', 'data', allow_duplicate=True),
        Input('train-mlp', 'n_clicks'),
        State('input_num_problematic', 'value'),
        prevent_initial_call=True        
    )
    def train_and_predict(n_clicks,num_problematic):
        """
        Important function in which a MLP classifier is trained to predict the classes
        of unannotated data points, based on user annotation data. Is also responsible
        for highlighting difficult tracks with ‼️ (CPU) and ↔️ (DC), or both ‼️↔️ (CPU & DC),
        based on the heuristics implemented for the Master's thesis. 
        """
        # Start process time measurement
        s_time = time.time()

        # Error handling
        if n_clicks == 0:
            raise PreventUpdate
        
        # Error handling
        if data_store["df"] is None or data_store["df"].empty or not data_store["assigned_classes"]:
            return dash.no_update

        # Retrieve KFold parameters from config.py
        kfold_value = config["default_kfold"]

        # Retrieve number of highlighted candidate tracks for reannotation from config.py
        num_problematic = num_problematic if num_problematic is not None else config["default_problematic_tracks_num"]
        
        # Initialise lists and store labeled points and corresponding classes
        X_labeled = []
        Y_labeled = []
        for idx, label in data_store["assigned_classes"].items():        
            X_labeled.append(data_store["embeddings"][idx])
            Y_labeled.append(label)

        # Error handling
        if len(X_labeled) < 2:
            return dash.no_update
        
        # If calculation of KFold is set to True in config.py
        if config["calculate_kfold"]:
            # Iteratively train MLP classifier on expanding training set with KFold 
            # TO DO: ensure that training set in KFold is expanded
            kfold_scores = mh.compute_kfold_scores(X_labeled=X_labeled, Y_labeled=Y_labeled, k=kfold_value)
        else:
            # Return generic score value when KFold is not activated 
            kfold_scores = [-1]

        # Train MLP classifier on full set of labeled X and Y
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(100,), max_iter=500, random_state=1)
        clf.fit(X_labeled, Y_labeled)        

        # Retrieve all class labels 
        all_classes = clf.classes_

        # Initialise lists/dictionaries 
        predicted_classes = {}
        predicted_probabilities = {}
        probability_strings = {}
        is_problematic_per_track = {}
        unlabeled_distances = []
        min_distances_to_annotated = []
        probability_deltas = []

        # Initialise helper variables
        assigned_indices = set(data_store["assigned_classes"].keys())
        embeddings = data_store["embeddings"]

        # IMPORTANT: Candidate selection for reannotation based on CPU/DC heuristic 
        # Problematic tracks get highlighted in this loop
        for i, embedding in enumerate(embeddings):
            # Use trained MLP to retrieve predicted class and associated probabilities
            predicted_class, probabilities = uh.predict_class_and_probs(clf=clf, embedding=embedding)
            # Format probabilities as string
            prob_string = mh.format_prob_string(probabilities,all_classes,num_decimals=7)

            # Store retrieved predicted class/probabilities/formatted probability string
            predicted_classes[i] = predicted_class
            predicted_probabilities[i] = probabilities
            probability_strings[i] = prob_string

            # IMPORTANT: Calculate minimum (Euclidean) distance to closest annotated point (DC)
            # and minimum delta between two highest predicted probabilities (CPU)
            if i not in assigned_indices:                
                # IMPORTANT: DC Heuristic
                min_dist = uh.min_cosine_distance_to_annotated(i=i, embeddings=embeddings, assigned_indices=assigned_indices)
                # IMPORTANT: CPU Heuristic
                prob_delta = uh.return_top_two_probability_delta(probabilities.tolist())

                # Store DC/CPU values
                min_distances_to_annotated.append(min_dist)
                probability_deltas.append((i, prob_delta))
                unlabeled_distances.append((i, min_dist))
            else:
                # Annotated points are not considered for CPU/DC heuristic
                actual_class = data_store["assigned_classes"][i]
                track_status = uh.labeled_track_status(predicted_class, actual_class)

                is_problematic_per_track[i] = track_status
                min_distances_to_annotated.append(-1)

        # Get top N problematic indices associated with highlighted (DC/CPU) tracks 
        unlabeled_distances.sort(key=lambda x: x[1], reverse=True)
        top_problematic_indices = {idx for idx, _ in unlabeled_distances[:num_problematic]}

        probability_deltas.sort(key=lambda x: x[1])
        top_probability_problematic_indices = {idx for idx, _ in probability_deltas[:num_problematic]}

        # Assign track status to unannotated data points
        for i in range(len(embeddings)):
            if i not in assigned_indices:
                track_status = ""
                # # If data point is marked as problematic by CPU heuristic
                if i in top_probability_problematic_indices:
                    track_status += "‼️"
                # If data point is marked as problematic by DC heuristic
                if i in top_problematic_indices:
                    track_status += "↔️"

                # Store track status 
                is_problematic_per_track[i] = track_status

        # Update data_store
        data_store["predicted_classes"] = predicted_classes
        data_store["predicted_probabilities"] = predicted_probabilities
        data_store["probability_strings"] = probability_strings
        data_store["is_problematic_tracks"] = is_problematic_per_track
        data_store["accepted_predictions"] = False
        data_store["cosine_distances"] = min_distances_to_annotated
        data_store["training_completed"] = n_clicks
        df = uh.update_dataframe(df=data_store["df"],data_store=data_store,include_predictions=True)
        data_store["df"] = df
        
        # Predicate data point size on size_selection param in config.py
        size_selection = config["size_selection"]
        size_column = "BPM_size" if size_selection == 'BPM' else "Embedding_variance_size"

        # Create figure (2D scatter plot)
        fig = uh.create_figure(df = df, size_column = size_column, accepted = False) 
        
        # Update datatable
        predicted_classes_copy = dict(data_store["predicted_classes"])
        table_data = uh.create_table_data(predicted_classes=predicted_classes_copy,ids=data_store["ids"],config=config,data_store=data_store,table_type="predicted")
        
        # Stop process time measurement
        e_time = time.time()


        event_info = {
            "Timestamp": str(datetime.datetime.now().isoformat()),
            "Event Type": "Train MLP and Predict Classes",
            "Event Process Time": str(round(e_time - s_time, 5)),
            "MLP Training Iteration": str(data_store["iteration"]),
            "MLP Classifier Scores": kfold_scores,
            "MLP Classifier Average Score": str(np.mean(kfold_scores)),
            "Predicted Classes": {str(k): v for k, v in predicted_classes.items()}  # Convert keys to strings for JSON compatibility

        }

        data_store["iteration"] += 1

        json_log["Events"].append(event_info)

        return fig,table_data,{'trained': True, 'timestamp': time.time()}

    # ----------------------------------- ACCEPT/REJECT PREDICTIONS -----------------------------------

    @app.callback(
        Output('umap-plot', 'figure', allow_duplicate=True),
        Input('accept-classifications', 'n_clicks'),
        prevent_initial_call=True
    )
    def accept_classifications(n_clicks):
        """
        Function which updates the UI state when user accepts the MLP classifier
        predictions. Ensures that all data points are set to 1.0 opacity in the 
        2D plot after predictions are accepted. 
        """
        # Start process time measurement
        s_time = time.time()

        # Read out which action triggered this function
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

        # Error handling
        if data_store["df"] is None or not data_store["predicted_classes"]:
            return dash.no_update
        
        # Combine all data points; "predicted_classes" with "assigned_classes" 
        for idx, predicted_class in data_store["predicted_classes"].items():
            if idx not in data_store["assigned_classes"]:  
                data_store["assigned_classes"][idx] = predicted_class
        
        # Mark predictions as accepted
        data_store["accepted_predictions"] = True

        # Update DataFrame with all classes now at full opacity
        df = data_store["df"].copy()
        for i in range(len(df)):
            df.loc[i, "Class"] = data_store["assigned_classes"].get(i, "?")
            df.loc[i, "Opacity"] = 1.0  # Set opacity to 1.0 to signify accepted predictions
        
        # Store updated DataFrame
        data_store["df"] = df

        # Predicate data point size on size_selection param in config.py
        size_selection = config["size_selection"]
        size_column = "BPM_size" if size_selection == 'BPM' else "Embedding_variance_size"

        # Create figure (2D scatter plot)
        fig = uh.create_figure(df = df, size_column = size_column, accepted = True) 

        # Stop process time measurement
        e_time = time.time()

        event_info = {
            "Timestamp": str(datetime.datetime.now().isoformat()),
            "Event Type": "Classifications Accepted",
            "Event Process Time": str(round(e_time - s_time, 5)),
            "Triggered Dash ID": str(triggered_id)        
        }

        json_log["Events"].append(event_info)

        return fig
    
    @app.callback(
        Output('umap-plot','figure',allow_duplicate=True),
        Input('reject-predictions','n_clicks'),
        prevent_initial_call=True
    )
    def reject_predictions(n_clicks):
        """
        Function which reverts back to the UI state in which data points
        are either annotated by the user, or not annotated yet (? as class name)
        """
        # Start process time measurement
        s_time = time.time()

        # Error handling
        if data_store["df"] is None:
            return dash.no_update
        
        # Clear predicted classes
        data_store["predicted_classes"] = {}
        data_store["accepted_predictions"] = False
        
        # Update df
        df = uh.update_dataframe(df=data_store["df"],data_store=data_store,include_predictions=False)
        data_store["df"] = df    

        # Predicate data point size on size_selection param in config.py
        size_selection = config["size_selection"]
        size_column = "BPM_size" if size_selection == 'BPM' else "Embedding_variance_size"

        # Create figure (2D scatter plot)
        fig = uh.create_figure(df = df, size_column = size_column, accepted= False) 
        
        # Stop process time measurement
        e_time = time.time()

        event_info = {
            "Timestamp": str(datetime.datetime.now().isoformat()),
            "Event Type": "Classifications Rejected",
            "Event Process Time": str(round(e_time - s_time, 5)),
            # "Triggered Dash ID": str(triggered_id)        
        }

        json_log["Events"].append(event_info)
        
        return fig

    # ----------------------------------- HELPER/UTILITY FUNCTIONS (AUDIO) -----------------------------------

    @app.callback(
        Input("audio-path",'value')
    )
    def set_audio_path(input_audio_path):
        """
        Updates config.py to store audio path provided by user
        """
        config["audio_path"] = input_audio_path


    @app.callback(
        Input("audio-play-select",'value')
    )
    def set_audio_play(input_audio_play_selection):
        """
        Updates config.py to determine if audio is played back on hover
        """
        if input_audio_play_selection == "Yes":
            config["play_audio"] = True
        else:
            config["play_audio"] = False


    @app.callback(
        Output('hover-audio-path', 'data'),
        Input('umap-plot', 'hoverData'),
        prevent_initial_call=True
    )
    def handle_hover(hoverData):
        """
        Function which handles audio playback on hover. 
        Is called when the user hovers over a data point in the
        2D embedding space plot. Returns audio path if audio playback
        on hover is active. 
        """
        # Error handling
        if not hoverData or 'points' not in hoverData:
            return dash.no_update

        # Get data point which is hovered on by user
        point = hoverData['points'][0]
        idx = point['customdata'][0]  # NOTE: Assumes ID is at index 0 of customdata

        # If audio playback is set to on, locate the audio file in audio 
        # file directory provided by user
        if config["play_audio"]:
            audio_path = uh.find_audio_file(config=config,music_id=idx)
        else:
            audio_path = None

        # Function does not update if audio file path is not found
        if not audio_path:
            return dash.no_update

        # Initialise temporary local audio path for audio playback
        web_path = f"/audio/{os.path.relpath(audio_path, config['audio_path'])}"
        
        return {'audioPath': web_path}

    # Serve audio files
    @app.server.route('/audio/<path:path>')
    def serve_audio(path):    
        """
        Function which returns relevant audio file as a HTTP response, 
        retrieved automatically from audio file directory provided by user.
        """
        full_path = os.path.join(config['audio_path'], path)
        try:
            return send_file(full_path, mimetype='audio/mpeg') 
        except:
            return "File not found", 404

    # ----------------------------------- HELPER/UTILITY FUNCTIONS (MISC) -----------------------------------

    @app.callback(
        Output("download-annotated", "data"),
        Input("download-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_results(n_clicks):
        """
        Function which formats annotated data and MLP classifier predictions
        into a formatted and downloadable .txt file. Returns .txt file when 
        download button is clicked by user.
        """
        # Error handling
        if n_clicks is None or n_clicks == 0:
            return dash.no_update

        # .txt file section 1: Introduction
        annotated_lines = ["------- User annotations and ML predictions for unannotated data -------\n"]
        annotated_lines.append("In this .txt file, you can find the points which you have annotated, and the points which have been predicted by the ML classifier. The data is displayed in columnar format; \n \n For the annotated data, the columns are [Song ID, Assigned Class].\n\n For the predictions, the columns are [Song ID, Predicted Class, Prediction Probabilities per Class]. \n\n--------------------------------------------------------------------------- \n")

        # .txt file section 2: Annotated data 
        annotated_lines.append("Annotated Songs [Song ID, Assigned Class]:\n\n")
        for i, cls in data_store["assigned_classes"].items():
            if i < len(data_store["ids"]):
                annotated_lines.append(f"{data_store['ids'][i]}\t{cls}")
        
        # .txt file section 3: Predicted data (ID, Predicted Class, Probabilities)
        predicted_lines = ["\nPredictions [Song ID, Predicted Class, Prediction Probabilities per Class]:\n\n"]
        predicted_classes = data_store.get("predicted_classes", {})
        for idx, pred_class in predicted_classes.items():
            if idx < len(data_store["ids"]):
                prob = data_store["probability_strings"][idx] if data_store["probability_strings"] is not None else "N/A"
                predicted_lines.append(f"{data_store['ids'][idx]}\t{pred_class}\t{prob}")
        
        content = "\n".join(annotated_lines + predicted_lines)

        return dict(content=content, filename="classification_results.txt", type="text/plain")

    # Function for resetting data table and statistics 
    @app.callback(
        [Output('selected-data-table', 'data'),
        Output('predicted-data-table', 'data', allow_duplicate=True),
        Output('problematic-tracks-stats', 'children', allow_duplicate=True)],
        [Input('reset-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def reset_data_tables_and_stats(reset_clicks):
        """
        Function which clears both data tables, and resets problematic tracks 
        statistics when the reset button is clicked by the user
        """
        if reset_clicks:
            # Return empty lists for both tables, reset message for stats
            return [], [], "Statistics will appear after training the classifier"
        
        # Error handling 
        return [], [], "Statistics will appear after training the classifier"


    @app.callback(    
        Output('download-json','data'),
        Input('json-log-radio-items','value'),
        Input('download-json-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_json_data(choice,n_clicks):
        """
        NOTE: this function should be expanded in a later version. 
    
        Function which 'jsonifies' all user action data which is stored
        in the json_log variable. The json_log variable is responsible 
        for storing user action data within the UI (timestamp, user action, etc.)
        """

        if choice == "Verbose" and n_clicks > 0:
            verbose_jsonified = jh.jsonify_log(copy.deepcopy(json_log))        

            print("TESTING LOG\n")
            print(verbose_jsonified)
            print("TESTING LOG")

            with open('logs.json','w') as fp:
                json.dump(verbose_jsonified,fp)

            return dcc.send_file('logs.json')
        if choice == "Stats" and n_clicks > 0:
            stats_jsonified = jh.jsonify_log(copy.deepcopy(stats_log))

            with open('stats.json','w') as fps:
                json.dump(stats_jsonified,fps)

            return dcc.send_file('stats.json')

    @app.callback(
        Output('problematic-tracks-stats', 'children'),
        [Input('training-state', 'data')],  # Use training-state as a trigger
        prevent_initial_call=True
    )
    def update_problematic_tracks_stats(training_state):
        """
        Function which updates the problematic tracks statistics section
        """
        # While not all data has been computed display "MLP classifier training..."
        if (not data_store.get("predicted_classes") or 
            not data_store.get("is_problematic_tracks") or 
            len(data_store.get("is_problematic_tracks", {})) == 0):
            return "MLP classifier training..."
        
        # Return formatted problematic tracks stats section
        stats_content = th.format_track_stats_component(data_store=data_store)
        return stats_content


    @app.callback(
        Output('training-state', 'data'),
        Input('train-mlp', 'n_clicks'),
        prevent_initial_call=True
    )
    def update_training_state(n_clicks):
        if n_clicks > 0:
            return {'timestamp': time.time(), 'n_clicks': n_clicks}
        return dash.no_update
