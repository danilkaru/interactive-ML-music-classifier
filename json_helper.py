# Helper file containing functions related to JSON functionality in the UI

import json
import base64
import numpy as np
import pandas as pd

def process_json(contents):
    """
    Function which extracts embeddings, ids and BPM values from an uploaded
    .json embedding file.
    NOTE: If an audio file has multiple embedding vectors, the embedding 
    vectors are averaged out to a single embedding vector. 
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')
    
    try:
        # Load embedding .json file
        embeddings_data = json.loads(decoded)
        
        # Extract embeddings and IDs from uploaded .json file
        embeddings = []
        ids = []
        bpm_values = []

        for entry in embeddings_data:
            embedding = np.array(entry["embedding"])
            if len(embedding.shape) > 1:  # If it's multi-dimensional (e.g., [T, 768])
                embedding = embedding.mean(axis=0)  # average out the embedding to a single embedding vector
            embeddings.append(embedding)
            
            # Extract ID
            music_id = entry.get("id", "")
            ids.append(music_id)

            # Extract BPM
            bpm = entry.get("bpm", 1)
            bpm_values.append(bpm)
        
        # Convert to NumPy array
        embeddings = np.vstack(embeddings)
        return embeddings, ids, bpm_values
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None, None, None
        

def jsonify_log(obj):
    """
    Function which (recursively) 'jsonifies' the json_log variable from app_callbacks.py
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = jsonify_log(value)
        return obj
    elif isinstance(obj, list):
        return [jsonify_log(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_json()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj
