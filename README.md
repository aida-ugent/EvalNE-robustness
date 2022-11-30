# Code for reproducing the experients of "A Systematic Evaluation of Node Embedding Robustness", LoG 2022 #

This folder contains the EvalNE framework as well as our proposed extension towards robustness 
evaluation. This code together with the config files included in the `experiments` folder
can be used to replicate the experimental results reported in our manuscript. See the 
Experiments section below for more details.

## Installation ##

The library requires Python 2.7. Other package dependencies are listed in the 
`requirements.txt` file.

Before installing the library, ensure that some necessary packages are present:
```bash
sudo apt-get install python-pip
sudo apt-get install python-tk
```

Install the requirements and the EvalNE package:
```bash
pip install -r requirements.txt
sudo python setup.py install
```

## Experiments ##

In order to replicate our experiments we provide 4 EvalNE configuration files. Below we show how 
to use these config files and later explain the relations between them and the experiments in 
our manuscript.

Running an evaluation in EvalNE with a given config file simply requires the file path to be 
provided as a parameter, e.g.:
```bash
python -m evalne ./experiments/exp_nc_rand.ini
```

Config files are named based on the downstream task and poison attack type, thus, we have:
 * Random attacks on node classification experiment: `exp_nc_rand.ini`
 * Adversarial attacks on node classification experiment: `exp_nc_adv.ini`
 * Random attacks on network reconstruction experiment: `exp_nr_rand.ini`
 * Adversarial attacks on network reconstruction experiment: `exp_nr_adv.ini`
 
The evaluations must be run individually for each attack type and random seed. For instance,
one should run `exp_nc_rand.ini` once with attack strategy _add_edges_rand_ and seed _42_. 
Then the config file can be edited to change e.g. the attack type to _del_edges_rand_no_ and 
rerun the evaluation. In each config file we have added the attack strategies and seeds used 
as values that can be easily commented or uncommented when running the evaluations. See the 
config files for more details.


## Methods and Data ##

In order to simplify code and experiment checking we have included the data and methods used 
in the paper in two separate folder. 

**NOTE:** The methods should each be each installed individually in one virtual environment 
(preferably named venv) to ensure correct evaluation. Follow the instructions in the 
respective readmes for installation.
