# interactive-ML-music-classifier

# About
This repository contains the code for the user interface for IML-based (re-)classification of music collections, as part of the Master's thesis by Danila Alexandrovich Danilin for the Master in Sound and Music Computing, MTG UPF, Barcelona. 

# Setting up and running the UI
The UI can be installed on a local device by cloning into this Github repository. Python 3.10 must be installed on the local device to ensure proper functionality. After cloning into the repository, it is recommended to create a virtual environment for the installation of all necessary Python dependencies. After opening a command-line interface (CLI) and
changing to the directory in which the repository has been installed, you can run the following commands; 

```
conda create -n IMLMusic python=3.10 -y
conda activate IMLMusic
pip install -r requirements.txt
```

Subsequently, the UI software can be run from the CLI with;

```
python [DIRECTORY/TO/UI/REPO/user_interface.py]
```

The UI is now locally hosted and accessible with any browser on http://127.0.0.1:8050/.

# Generating audio embeddings
A .json file containing CLAP audio embeddings can be generated with the following Google Colab notebook: https://colab.research.google.com/drive/1s79vkd4nqH370V9KTwMDpQ8WiudtGlzG?usp=sharing. The resulting .json file can be uploaded in the UI.

# Interacting with the UI
After the UI has been succesfully activated, the page display should look as follows (formatting can vary depending on browser):
<img width="717" height="420" alt="UI_main" src="https://github.com/user-attachments/assets/2fdd7ff0-1852-438e-b8cc-8b9680f56b1f" />

## Uploading embeddings and activating audio playback
It is now possible to upload a .json embedding file and interact with the UI. In this repository, a .json MAEST embedding (Alonso-Jiménez et al., 2023) file has been included, generated from a 300-song subset of the MusAV dataset (Bogdanov et al., 2022). The embedding file can be found in the relative path interactive-ML-music-classifier/Embeddings
/embeddings_MusAV.json. 

The audio files of the 300-song subset of the MusAV dataset can be found on https://drive.google.com/drive/folders/1dmaGrrK7HqUINAqtUbXuJsaBmwwxBr6k?usp=drive_link. To ensure that audio playback on hover can be activated, download the audio file directory from the aforementioned URL, and then specify the path in the text input field next to "Play audio on hover?" (under the 2D plot). 

## Data visualisation and annotation
After the .json embedding file has been uploaded, a 2D visualization of the music collection is presented, derived from the music collection embedding space which has been reduced in dimensionality using the UMAP (McInnes & Healy, 2018) technique. Each audio is represented as a distinct data point in the interactive 2D plot. Audio files which are considered to be similar by the audio embedding model, are displayed close to each other in the 2D plot. 

To assign a class to one or more data points, hover on the 2D plot, and in the upper right-hand corner, a collection of symbols appears. Click on the "Box Select" or "Lasso Select" symbol, and then draw a boundary around the data points to be classified. Next, in the upper left-hand corner, enter a class name which describes the selected data points. Then, click the "Assign Class" button.  

The plot should now consist of at least one annotated region, such as in the following example:

<img width="716" height="247" alt="Screenshot 2025-08-01 at 16 00 15" src="https://github.com/user-attachments/assets/a8ec5172-46d9-4977-a9f6-422bb8b31a65" />

**NOTE:** It is possible to change the shape of the data 'cloud' by experimenting with different values for the "UMAP_seed" parameter in config.py (ui_layout_config dictionary). This will produce different 2D representations of the music collection, which may be useful for discovery and exploration purposes. By default, the "UMAP_seed" value is set to 7.

## Classifier training and predictions
After all relevant data point regions have been assigned with any class of choice, a multilayer perceptron (MLP) classifier can be trained to predict the class of each remaining unannotated data point. To train the MLP classifier, click on the "Train & Predict" button. After the classifier is trained, predictions are displayed as follows: 

<img width="707" height="197" alt="Screenshot 2025-08-01 at 17 34 53" src="https://github.com/user-attachments/assets/fcc54232-17f1-473c-8808-2054b38f4917" />

Some tracks are highlighted with additional symbols; tracks with the ‼️ symbol have high class prediction uncertainty (CPU heuristic), and tracks with the ↔️ symbol have a high (Euclidean) distance to the nearest annotated data point (DC heuristic). Note that for the ↔️ symbol/DC heuristic, the mutual Euclidean distance is not calculated with the 2D coordinates of data points, but using the full, higher-dimensional embedding vectors associated with an audio file. 

By adding the highlighted tracks to the training set in future classifier training iterations, it is hypothesized that the classifier is able to reach a higher classification accuracy within fewer training iterations. This is discussed in more detail in the thesis manuscript. 

More details on the classifier predictions can be found in the "Problematic Tracks Statistics" component of the UI: 

<img width="714" height="234" alt="Screenshot 2025-08-01 at 17 38 43" src="https://github.com/user-attachments/assets/8ef5cdf9-caa0-4fa3-b129-c0db16b9e451" />

Furthermore, the datatables contain a detailed overview of track status, classifier prediction probabilities and other metadata:

<img width="713" height="341" alt="Screenshot 2025-08-01 at 17 40 01" src="https://github.com/user-attachments/assets/4d94e8b3-1011-438c-b434-2bf19c1c7f14" />


## Accepting classifier predictions
When, in your opinion, the classifier is good enough at predicting classes for unannotated data points, the classifications can be accepted with the "Accept Predictions" button. 

To save the classifier predictions, the "Download results" in the bottom-right corner can be clicked. A .txt file containing user class annotations and classifier predictions for unannotated data points is then downloaded. 


# Academic sources mentioned in this repo
* Alonso-Jiménez, P., Serra, X., & Bogdanov, D. (2023). Efficient supervised training of audio transformers for music representation learning. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2309.16418
* Bogdanov, D., Lizarraga-Seijas, X., Alonso-Jiménez, P., & Serra X. (2022). MusAV: A dataset of relative arousal-valence annotations for validation of audio models. International Society for Music Information Retrieval Conference (ISMIR 2022). http://hdl.handle.net/10230/54181
* McInnes, L., & Healy, J. (2018). UMAP: uniform manifold approximation and projection for dimension reduction. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1802.03426

