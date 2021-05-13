# Part 2 Project

Contains code for my Part 2 Project which allows the training, testing and attacking of model. Functionality for adversarial training using any one of the attacks. The 4 adversarial attacks implemented are:
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Carlini-Wagner (CW) L2
- CW L-infinity

## Training a model

From the bin folder run e.g.:
```
python train.py --train-acc [file to write train accuracy] --val-acc [file to write validation accuracy]
--train-loss [file to write train loss]  --val-loss [file to write validation loss] 
-p [path to save model]
```

### Plotting the train, validation loss and accuracy

From the plotting folder run e.g.:

```
python plot.py --train-acc [path to train accuracy file] --val-acc [path to val accuracy file]
--train-loss [path to train loss file] --val-loss [path to val loss file]
--acc-filename [name of accuracy plot] --loss-filename [name of loss plot]
```

## Testing a model

From the bin folder run e.g.:
```
python test.py -p [path to model to be tested]
```

## Running Attacks

From the attacks folder run e.g.:
```
python attack.py -p [path to model to be attacked] -a [attack name - either fgsm, pgd, cwl2 or cwlinf]
```

Optional parameters:
- `-f [name of accuracy vs epsilon plot file]`
- `-af [name of file for saving 5 example adversarial images]`
- `-of [name of file for saving the 5 original images]`
- `-pf [name of file for saving the example perturbations produced]`
- `-mp [True/False]` - True to magnify perturbations 100x

## Adversarial Training

From the bin folder run e.g.:
```
python train.py --train-acc [file to write train accuracy] --val-acc [file to write validation accuracy]
--train-loss [file to write train loss]  --val-loss [file to write validation loss] 
-p [path to save model] -a [name of attack (either fgsm, pgd, cwl2 or cwlinf)]
```
