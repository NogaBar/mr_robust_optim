# Multiplicative Reweighting for Robust Optimization

Use multiplicative weight algorithm for reweighting examples for robust neural network optimization.

### Installation
Install the dependencies:
~~~~
pip install -r requirements.txt
~~~~
### Training
For training use `main.py` file. The repository supports CIFAR10/100.
Example for training command line:
~~~~
python main.py --dataset cifar10 --batch-size 128 --weight-update mr --checkpoint-dir cifar10_mr
~~~~ 

