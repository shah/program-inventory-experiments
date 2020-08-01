# Program Inventory Experiements

This repo contains code for ML and NLP experiments to help with auto classification of government programs that 
may be referenced in legislative and budget narrative text.

## Prerequisites

* Docker Desktop (latest version) on any supported platform
* Visual Studio Code (latest version) on any supported platform, using .devcontainer model. 
* Visual Studio Code Remote Containers extension enabled

## Instructions

* Start Docker Desktop
* `cd <your_projects_directory>`
* `git clone https://github.com/shah/program-inventory-experiments.git`
* `cd program-inventory-experiments`
* `code .` (this starts Visual Studio Code)
* Reload the folder in Remote Container when suggested by VS Code
* Open `train.py` in VS Code editor and see which models you'd like to train, then save the file
* In the VS Code Terminal:
  * `python train.py`
* After that finishes:
  * The trained models will be placed in `./models-trained`
  * Open `predict.py` in VS Code editor (just for reference)
  * Open `predict-in-legislative.csv` in VS Code editor and add some test cases
  * In the VS Code Terminal run `python predict.py`
* Open `predict-out-legislative.csv` to see predictions
