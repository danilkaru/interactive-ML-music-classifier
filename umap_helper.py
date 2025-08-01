# Helper file containing important function for applying UMAP dimensionality reduction
# on a list of embeddings, necessary for producing a 2D visualization of the music 
# collection embedding space. 

# This file also contains other miscellaneous helper functions which will be moved to 
# other (helper) files in future updates. 


import umap
import plotly.express as px
import numpy as np
from scipy.spatial.distance import cosine
import os
from config import config, ui_layout_config

# Function to compute UMAP
def apply_umap(embeddings):
    """
    Function which applies UMAP dimensionality reduction on provided list of 
    embeddings. Essential for the 2D visualization of the music collection 
    embedding space. 
    """

    # Initialise UMAP variables
    # TO DO: add as configuration parameters to config.py
    n_neighbors = 15
    min_dist = 0.1
    metric = "cosine"
    random_state = ui_layout_config["UMAP_seed"]

    # Store UMAP state in json_log (future versions)
    UMAP_data = {}
    UMAP_data.update({'n_neighbors':n_neighbors,'min_dist':min_dist,'metric':metric,'random_state':random_state})    
    # json_log.update({'UMAP':UMAP_data})

    # Apply UMAP on embeddings 
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)

    # Return UMAP representation of embeddings
    return reducer.fit_transform(embeddings), UMAP_data

def create_figure(df, size_column, accepted = False):
    """
    Function for creating the 2D scatter plot used for visualizing
    the music collection embedding space
    """
    # Different plot when predictions are accepted
    if accepted:
        fig_text = "Class"
    else:
        fig_text = "ClassDisplay"

    # Create plot
    fig = px.scatter(df, x='x', y='y', text=fig_text, color='Class', 
                     size=size_column, opacity=df["Opacity"],
                     hover_data=["ID", "IsProblematic", "Probabilities"],
                     title="UMAP Projection", custom_data=["ID", "IsProblematic", "Probabilities"])
    
    return fig
    
def scale_values(values, offset, scale):
    """
    Function for scaling a list of values, and shifting them by some offset value
    """
    if values.max() > 1:
        values_scaled = offset + (values-values.min()) / (values.max() - values.min()) * scale
    else:
        values_scaled = np.full(len(values),offset) 
    return values_scaled


def predict_class_and_probs(clf, embedding):
    """
    Function for retrieving MLP classifier predictions and associated probabilities
    """
    predicted_class = clf.predict([embedding])[0]
    probabilities = clf.predict_proba([embedding])[0]
    return predicted_class, probabilities

def min_cosine_distance_to_annotated(i, embeddings, assigned_indices):
    """
    Function for calculating the minimum distance to the nearest annotated data point
    Part of the DC heuristic
    """
    return min(
        abs(cosine(embeddings[i], embeddings[j]))
        for j in assigned_indices
    )


def labeled_track_status(predicted,actual):
    """
    Function for displaying track status of user annotated data points
    If the predicted class doesn't equal the annotated class, ❔ is displayed
    """
    return "(User Annotation) ✅" if predicted == actual else "(User Annotation) ❔"


def find_audio_file(config,music_id):
    """
    Function for retrieving the audio file path for a provided song/music ID
    """
    if not music_id:
        return None
    for root, dirs, files in os.walk(config["audio_path"]):
        for file in files:
            # if file.startswith(music_id) and file.endswith('.mp3'):
            if file.startswith(music_id):
                path = os.path.join(root, file)
                print(f"Found audio file at: {path}")  
                return path
    
    print(f"Audio file not found for ID: {music_id}")  
    return None


def return_top_two_probability_delta(prob_list):
    """
    Function for calculating the delta between the two highest probabilities 
    in a list of probabilities. 
    Part of the CPU heuristic
    """
    # Error handling
    if len(prob_list) < 2:
        return "Error"

    # Sort probabilities list  
    sorted_probs = sorted(prob_list,reverse=True)

    # Return delta value
    return abs(sorted_probs[0] - sorted_probs[1])


def create_table_data(predicted_classes, ids, config, data_store, table_type):
    """
    Function for aggregating all necessary data to be displayed in the datatables
    TO DO: rename predicted_classes parameter name! This function is used for 
    both the user annotation datatable and the prediction datatable
    """
    # Initialise table_data 
    table_data = []

    # Loop over annotated/prediction data points
    for i, cls in predicted_classes.items():

        # Create datatable for annotated data points
        if i < len(ids):
            if table_type == "annotated" or i not in data_store["assigned_classes"]:
                music_id = ids[i]
                if config["play_audio"]:
                    audio_path = find_audio_file(config=config,music_id=music_id)
                else:
                    audio_path = None

                web_path = f"/audio/{os.path.relpath(audio_path, config['audio_path'])}" if audio_path else "Not found"
                audio_button = f'<button onclick="playAudio(\'{i}\', \'{web_path}\')">Play</button>' if audio_path else "Not found"

                audio_entry = {
                    "Index": i,
                    "ID": music_id,
                    "Class": cls,
                    "AudioPath": web_path if audio_path else "Not found",
                    "Audio": audio_button,                    
                }

                if table_type == "annotated":
                    audio_entry["TrackStatus"] = "👤"

                # # Create datatable for predicted data points
                else:
                    audio_entry["Probabilities"] = str(data_store["probability_strings"][i])
                    audio_entry["CosineDistanceToCentroid"] = float(data_store["cosine_distances"][i])
                    if str(data_store["is_problematic_tracks"][i]) in ["‼️", "↔️", "‼️↔️"]:
                        audio_entry["TrackStatus"] = str(data_store["is_problematic_tracks"][i])
                    else:
                        audio_entry["TrackStatus"] = "✅"

                table_data.append(audio_entry)

    return table_data


def extract_display_symbols(track_status):
    """
    Function for extracting symbols from track_status 
    """
    symbols = ""
    if "‼️" in track_status:
        symbols += "‼️"
    if "↔️" in track_status:
        symbols += "↔️"
    if "❔" in track_status:
        symbols += "❔"
    return symbols





def update_dataframe(df, data_store, include_predictions=False):
    """
    Function for updating df data in data.py
    """
    # Create copy of df
    df = df.copy()

    for i in range(len(df)):
        # Case 1: Data point is annotated by user
        if i in data_store["assigned_classes"] and data_store["assigned_classes"][i] is not None:
            # Manual assignment (always shown)
            cls = data_store["assigned_classes"][i]
            df.loc[i, "Class"] = cls
            df.loc[i, "ClassDisplay"] = f"👤{cls[:2]}" if cls else "?"
            df.loc[i, "Opacity"] = 0.2
            
            if include_predictions:
                # When predictions are included, show actual probability/status data
                df.loc[i, "Probabilities"] = data_store.get("probability_strings", {}).get(i, "👤 (user annotation)")
                df.loc[i, "IsProblematic"] = data_store.get("is_problematic_tracks", {}).get(i, "👤 (user annotation)")
            else:
                # When predictions are rejected, show user annotation labels
                df.loc[i, "Probabilities"] = "👤 (user annotation)"
                df.loc[i, "IsProblematic"] = "👤 (user annotation)"

        # Case 2: Data point is unannotated, but is in "predicted_classes" (include_predictions=True)
        elif include_predictions and i in data_store.get("predicted_classes", {}):            
            track_status = data_store.get("is_problematic_tracks", {}).get(i, "")
            display_symbols = extract_display_symbols(track_status)  # Excludes ✅
            
            cls = data_store["predicted_classes"][i]
            df.loc[i, "Class"] = cls
            df.loc[i, "ClassDisplay"] = f"{cls[:2]}{display_symbols}" if cls else "?"
            df.loc[i, "Opacity"] = 1.0 if data_store.get("accepted_predictions", False) else 0.2
            df.loc[i, "Probabilities"] = data_store.get("probability_strings", {}).get(i, "N/A")
            df.loc[i, "IsProblematic"] = str(track_status)
            
        # Case 3: Initialising df (include_predictions=False)
        else:
            df.loc[i, "Class"] = "?"
            df.loc[i, "ClassDisplay"] = "?"
            df.loc[i, "Opacity"] = 0.2
            df.loc[i, "Probabilities"] = "N/A"
            df.loc[i, "IsProblematic"] = "N/A"

    return df