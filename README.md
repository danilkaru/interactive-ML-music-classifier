# interactive-ML-music-classifier

# About
This repository contains the code for the user interface for IML-based (re-)classification of music collections, as part of the Master's thesis by Danila Alexandrovich Danilin for the Master in Sound and Music Computing, MTG UPF, Barcelona.

# Setting up the UI
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

