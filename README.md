# pytorch-SIGUA
SIGUA (Stochastic Integrated Gradient Underweighted Ascent) is a robust optimization technique for training models in label noises. This method applies gradient descent on good data (possibly clean data) as usual, and learning-rate-reduced gradient ascent on bad data (possibly noisy data).

Paper: https://proceedings.icml.cc/static/paper_files/icml/2020/705-Paper.pdf

### Usage
0. Install requirements.txt
~~~
pip install -r requirements.txt
~~~

1. Preprocessing (build noisy data)
~~~
python main.py \
    --run_mode preprocess \
    --noise_prob 0.5 \
    --noise_type sym \
    --dataset MNIST
~~~

2. Training
~~~
python main.py \
    --run_mode train \
    --model sigua \
    --num_gradual 5 \
    --bad_weight 0.001 \
    --tau 0.5
    --lr 0.001 \
    --batch_size 256 \
    --num_class 10
~~~

### Experiments on MNIST (Image)

#### Performance results
* num_gradual = 5
* bad_weight = 0.001

| Settings / Models   	| CNN (reproduce, standard) 	| CNN (paper, standard) 	| CNN (reproduce, SIGUA) 	| CNN (paper, SIGUA) 	|
|---------------------	|:-------------------------:	|:---------------------:	|:---------------------------:	|:-----------------------:	|
| Sym (ε = 20%) 	|             98.3%              	|           -           	|          98.86%                   	|       98.91%                  	|
| Sym (ε = 50%) 	|       94.3%     	|         -       	|            98.38%            	|          98.10%          	|


#### Contact
Yeachan Kim (yeachan.kr@gmail.com)
