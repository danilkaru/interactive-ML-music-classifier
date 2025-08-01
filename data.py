# This file contains a data dictionary which is dynamically updated based on 
# user interaction with the user interface. The data dictionary is expanded
# with additional data in some of the callback functions in app_callbacks.py.

data_store = {
    "embeddings": None,
    "indeces": {},
    "df": None, 
    "assigned_classes": {}, 
    "colors": {}, 
    "ids": None, 
    "predicted_classes": {},
    "predicted_probabilities": {},
    "class_centroids": {},
    "is_problematic_tracks": {},
    "accepted_predictions": False,  # Flag to track if predictions have been accepted
    "training_time": str(0),
    "iteration":0
}