# pytorch-SIGUA
Pytorch implemetations of SIGUA

### Experiments on MNIST (Image)

#### Performance results
* num_gradual = 5
* bad_weight = 0.001

| Settings / Models   	| CNN (reproduce, standard) 	| CNN (paper, standard) 	| CNN (reproduce, SIGUA) 	| CNN (paper, SIGUA) 	|
|---------------------	|:-------------------------:	|:---------------------:	|:---------------------------:	|:-----------------------:	|
| Sym (ε = 20%) 	|             98.3%              	|           -           	|          98.86%                   	|       98.91%                  	|
| Sym (ε = 50%) 	|       94.3%     	|         -       	|            98.38%            	|          98.10%          	|

