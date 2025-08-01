# This file contains the configuration parameters for the user interface application

config = {
    # General
    "version": "0.0.1",
    "user": "user",
    "size_selection": "Embedding variance", #Options are "Embedding variance" and "BPM"

    # ML/heuristic related
    "calculate_kfold": True,
    "default_kfold": 5,
    "default_problematic_tracks_num": 20,
    
    # Audio related
    "play_audio": False,
    "audio_path": "/Users/danieldanilin/Downloads/master-thesis-full/musav-08-02-2025/audio_chunks"
}

# TO DO: expand/integrate with other config dictionaries
environment_variables = {
    
}


ui_layout_config = {
    # Data point sizes (in 2D graph)
    "UMAP_seed": 7,
    "size_offset": 10,
    "size_scale": 30,
}