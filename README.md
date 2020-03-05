dldo
==============================

A deep learning repo that tackles a discrete optimization problem.em

Project Organization
------------

    ├── LICENSE
    ├── README.md                               <- The top-level README for developers using this project.
    ├── data
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── docs                                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                                  <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                              the creator's initials, and a short `-` delimited description, e.g.
    │                                              `1.0-jqp-initial-data-exploration`.
    │
    ├── references                              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt                        <- The requirements file for reproducing the analysis environment, e.g.
    │                                              generated with `pip freeze > requirements.txt`
    │
    ├── dldo           <- Source code for use in this project.
    │   │
    │   ├── data                                <- Scripts to download or generate data
    │   │
    │   ├── utils                                <- Scripts utilities used during data generation or training
    │   │
    │   ├── training                            <- Scripts to train models
    │   │
    │   ├── validate                            <- Scripts to validate models
    │   │
    │   └── visualization                       <- Scripts to create exploratory and results oriented visualizations

Installation
------------
    
    conda create -n dldo python=3.7 
    conda install numpy scipy matplotlib seaborn pandas scikit-learn
    conda install ipykernel
    conda install pytorch torchvision -c pytorch
    conda config --add channels http://conda.anaconda.org/gurobi
    conda install gurobi
    
    # replace w/ your own license
    grbgetkey 7e5199fa-528f-11ea-9fde-0a7c4f30bdbe