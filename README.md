# ALTER: Augmentation for Large-Table-Based Reasoning
Official implementation of the paper "ALTER: Augmentation for Large-Table-Based Reasoning" 

## 📝 Paper

[ALTER: Augmentation for Large-Table-Based Reasoning]()

## 🚀 Installation
```
git clone https://anonymous.4open.science/r/tabular_data-295C
conda create -n alter python=3.10
conda activate alter
pip install -r requirements.txt
```

## 🧩Data

We use the following datasets for the experiments:

- [WikiTableQuestions](https://github.com/facebookresearch/WikiTableQuestions)

- [TabFact](https://github.com/allenai/TriviaQA)

## 🌲Main File Tree
#### Our Code

- [config.py](config): This directory contains the configs setting for the main experiment.

- [data](./data): This directory consists of python scripts to create PyTorch Dataset for different datasets and preprocessing utilities used for experiments

- [notebooks](./data): This directory contains python notebooks that were used carry out several ablations or run the experiment in one run on benchmarks.

- [prompt_manager](./prompt_manager): This directory mainly contains the prompt template to conduct experiments

- [utils](./utils): This directory contains the code for all the util tools code for ALTER workflow, such as normalization or parsing.

- [batch_pipe.py](./batch_pipe.py): This file is the main file for running the ALTER workflow.

```
 File Tree:
|-- ./augmentation.py # script for augmentation in pre-stage
|-- ./batch_pipe.py 
|-- ./data_loader/__init__.py
|-- ./data_loader/datasets
|-- ./data_loader/table_augmentation.py
|-- ./data_loader/table_format.py
|-- ./data_loader/TableLoader.py
|-- ./executor/executor.py
|-- ./run.py
`-- ./utils.py
|-- notebook
```