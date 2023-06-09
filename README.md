# MG_HEADSUP <img src="pics/headsup_logo.png" width="150px" align="right"/>
Hemmorrhage Evaluation And Detector System for Underserved Populations, a collaboration project with the joint efforts from Neurologists, and Machine Learning Scientist from the Mayo Clinic and Google.

[ArXiv]() | [Jounral Link]()

The Structure of the Codebase:
```
|- README.md
|- LICENSE
|- requirements.txt
|- pics
    |- headsup_logo.png
|- sample_data
    |- test.dcm
|- configs
    |- dataset.yaml
    |- preprocess.yaml
|- data_module
    |- dataset_prep.py
    |- img_preprocess.py
    |- util.py
|- notebook
    |- dataset_prep.ipynb
    |- dicom2png.ipynb
    |- channel_combine_2d.ipynb
    |- edge_detect.ipynb
    |- metadata_extract.ipynb
```
## Prerequisites:
```
conda create --prefix /path/to/conda/env_name python
pip3 install -r requirements.txt
```