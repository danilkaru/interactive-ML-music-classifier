# interactive-ML-music-classifier

# About
This repository contains the code for the user interface for IML-based (re-)classification of music collections, as part of the Master's thesis by Danila Alexandrovich Danilin for the Master in Sound and Music Computing, MTG UPF, Barcelona. 

The aim of this user interface is the identification of music tracks which 

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
python [DIRECTORY/TO/UI/REPO/user interface.py].
```

The UI is now locally hosted and accessible with any browser on http://127.0.0.1:8050/.

# Interacting with the UI
After the UI has been succesfully activated, the page display should look as follows (formatting can vary depending on browser):
<img width="717" height="420" alt="UI_main" src="https://github.com/user-attachments/assets/2fdd7ff0-1852-438e-b8cc-8b9680f56b1f" />

## Uploading embeddings and activating audio playback
It is now possible to upload a .json embedding file and interact with the UI. In this repository, a .json MAEST embedding (Alonso-Jiménez et al., 2023) file has been included, generated from a 300-song subset of the MusAV dataset (Bogdanov et al., 2022). The embedding file can be found in the relative path interactive-ML-music-classifier/Embeddings
/embeddings_MusAV.json. 

The audio files of the 300-song subset of the MusAV dataset can be found on https://drive.google.com/drive/folders/1dmaGrrK7HqUINAqtUbXuJsaBmwwxBr6k?usp=drive_link. To ensure that audio playback on hover can be activated, download the audio file directory from the aforementioned URL, and then specify the path in the text input field next to "Play audio on hover?" (under the 2D plot). 

## Data point annotation
After the .json embedding file has been uploaded, a 2D visualization of the music collection is presented, derived from the music collection embedding space which has been reduced in dimensionality using the UMAP (McInnes & Healy, 2018) technique. Each audio is represented as a distinct data point in the interactive 2D plot. 

To assign a class to one or more data points, hover on the 2D plot, and in the upper right-hand corner, a collection of symbols appears. Click on the "Box Select" or "Lasso Select" symbol, and then draw a boundary around the data points to be classified. Next, in the upper left-hand corner, enter a class name which describes the selected data points. Then, click the "Assign Class" button.  

The plot should now consist of at least one annotated region, such as in the following example:

<img width="716" height="247" alt="Screenshot 2025-08-01 at 16 00 15" src="https://github.com/user-attachments/assets/a8ec5172-46d9-4977-a9f6-422bb8b31a65" />




# Academic sources mentioned in this repo
* Alonso-Jiménez, P., Serra, X., & Bogdanov, D. (2023). Efficient supervised training of audio transformers for music representation learning. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2309.16418
* Bogdanov, D., Lizarraga-Seijas, X., Alonso-Jiménez, P., & Serra X. (2022). MusAV: A dataset of relative arousal-valence annotations for validation of audio models. International Society for Music Information Retrieval Conference (ISMIR 2022). http://hdl.handle.net/10230/54181
* McInnes, L., & Healy, J. (2018). UMAP: uniform manifold approximation and projection for dimension reduction. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1802.03426

