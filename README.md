# SMILES-RL
![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black) 

SMILES-RL is a framework for comparing reinforcement learning algorithms, replay buffers and scoring functions for SMILES-based molecular *de novo* design. The framework is originally developed for recurrent neural networks (RNNs) but can be used with other models as well.

## Prerequisites
Before you begin, ensure you have met the following requirements:

* Linux, Windows or macOS platforms are supported - as long as the dependencies are supported on these platforms.

* You have installed [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) with python 3.8 - 3.9, and poetry.

The tool has been developed on a Linux platform.

## Installation

First time, execute the following command in a console or an Anaconda prompt

    conda env create -f SMILES-RL.yml

This will install the conda environment. To activate the environment and install the packages using poetry
    
    conda activate SMILES-RL-env
    poetry install

## Usage

The framework currently provides a JSON-based command-line interface for comparing different RL algorithms, replay buffers and scoring functions. 


### Configuration File

To use the framework, a configuration file in [JSON format](https://en.wikipedia.org/wiki/JSON) which specifies all settings and absolute/relative paths to classes/constructors is needed. It should contain five main sections:
* **diversity_filter**: absolute/relative path to the diversity filter class (constructor) and corresponding parameters to use.
* **logging**:  absolute/relative path to the logger class and corresponding parameters to use.
*  **reinforcement_learning**: absolute/relative path to the reinforcement learning agent class and corresponding parameters to use.
*  **replay_buffer**: absolute/relative path to the replay buffer class and corresponding parameters to use.
*  **scoring_function**: absolute/relative path to the scoring function class (constructor) and corresponding parameters to use. 

Below is an example of such file using the diversity filter and scoring function of [REINVENT](https://github.com/MolecularAI/Reinvent), the **BinCurrent** replay buffer and the regularized maximum likelihood estimation agent (examples on scripts for generating configuration files are available in the directory `create_configs/` and see executable example below): 

```json
{
    "diversity_filter": {
        "method": "smiles_rl.diversity_filter.reinvent_diversity_filter_factory.ReinventDiversityFilterFactory",
            "parameters": {
                "name": "IdenticalMurckoScaffold", 
                "bucket_size": 25,  
                "minscore": 0.4,  
                "minsimilarity": 0.35,
            },
    },
    "logging": {
        "method": "smiles_rl.logging.reinforcement_logger.ReinforcementLogger",
        "parameters": {
            "job_id": "demo",
            "job_name": "Regularized MLE demo",
            "logging_frequency": 0,
            "logging_path": "progress_log",
            "recipient": "local",
            "result_folder": "results",
            "sender": "http://127.0.0.1"
        }
    },
    "reinforcement_learning": {
        "method": "smiles_rl.agent.regularized_mle.RegularizedMLE",
        "parameters": {
            "agent": "pre_trained_models/ChEMBL/random.prior.new",
            "batch_size": 128,
            "learning_rate": 0.0001,
            "n_steps": 2000,
            "prior": "pre_trained_models/ChEMBL/random.prior.new",
            "specific_parameters": {
                "margin_threshold": 50,
                "sigma": 128
            }
        }
    },
    "replay_buffer": {
        "method": "smiles_rl.replay_buffer.bin_current.BinCurrent",
        "parameters": {
            "k": 64,
            "memory_size": 1000
        }
    },
    "scoring_function": {
        "method": "smiles_rl.scoring.reinvent_scoring_factory.ReinventScoringFactory",
        "parameters": {
            "name": "custom_sum",
            "parallel": false,
            "parameters": [
                {
                    "component_type": "predictive_property",
                    "name": "classification",
                    "specific_parameters": {
                        "container_type": "scikit_container",
                        "descriptor_type": "ecfp_counts",
                        "model_path": "predictive_models/DRD2/RF_DRD2_ecfp4c.pkl",
                        "radius": 2,
                        "scikit": "classification",
                        "size": 2048,
                        "transformation": {
                            "transformation_type": "no_transformation"
                        },
                        "uncertainty_type": null,
                        "use_counts": true,
                        "use_features": true
                    },
                    "weight": 1
                }
            ]
        }
    }
}
```

Make sure to save this in JSON format, e.g., as `config.json`

### Running From the Command Line

After we have created and saved the configuration file, here saved as `config.json`, we can run it by

    python run.py --config config.json

### Note on Design of Reinforcement Learning Agent
The reinforcement learning agent, specified in section **reinforcement_learning** of the configuration file, takes an instance of scoring function, diversity filter, replay buffer or logger as input. The agent should use the given scoring function, diversity filter, replay buffer for update. Logger should be used for saving agent parameters and memory for intermediate and/or final inspection.

## Example
Below follows an executable example using a predictive model based on DRD2 data for activity prediction and which is utilizing a model pre-trained on the ChEMBL database. In this example we use proximal policy optimization (PPO) as reinforcement learning algorithm and all current (AC) as replay buffer. It uses the diversity filter and scoring function of [REINVENT](https://github.com/MolecularAI/Reinvent). Before preceding, make sure that you have followed the above installation steps. 

### Train Predictive Model
To create predictive model for activity prediction, run the training script

    python create_DRD2_data_and_models.py --fp_counts --generate_fps

which will train a random forest classifier saved to path `predictive_models/DRD2/RF_DRD2_ecfp4c.pkl`.

### Create Config File
Firstly, create logging directory where the configuration file and outputs will be saved

    mkdir logs

Run following script to create configuration JSON-file for PPO algorithm using all current as replay buffer, the ChEMBL pre-trained model and the predictive model trained above. 

    python create_configs/create_ppo_config.py --replay_buffer smiles_rl.replay_buffer.all_current.AllCurrent --log_dir logs --prior pre_trained_models/ChEMBL/random.prior.new --predictive_model predictive_models/DRD2/RF_DRD2_ecfp4c.pkl

### Run Experiment
Run experiment using the created configuration file

    python run.py --config ppo_config.json


## Current Agents and Replay Buffers
List of current agents and replay buffer combinations.

| RL Algo  | All Current        | Bin Current        | Top-Bottom Current | Top Current        | Top History        | Top-Bottom History | Bin History        |
| -------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| ACER     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A                | N/A                | N/A                |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Reg. MLE | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| SAC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A                | N/A                | N/A                |

## References
1. Hampus Gummesson Svensson, Christian Tyrchan, Ola Engkvist, Morteza Haghir Chehreghani. "Utilizing Reinforcement Learning for de novo Drug Design." ArXiv, (2023).
