# Multi-objects Generation with Amortized Structural Regularization
Multi-objects Generation with Amortized Structural Regularization

## Important Dependencies
    - Python 3.6.8
    - tensorflow-gpu 1.12.0
    - numpy 1.16.3

## Datasets
    - Use multi_mnist.py and multi_dsprites.py to synthesize.
    - Change the NUM_IN_COMMON in py files to change the number of objects, [1, 3] is the default setting for AIR-13, AIR-pPrior-13 and AIR-ASR 13.

    - For experiments on regularization on number of objects on Multi-MNIST: python multi_mnist.py --images-per-digit 20000 --name_surfix 20k
    - For experiments on regularization on number of objects on Multi-dSprites: python multi_dsprites.py --images-per-digit 20000 --name_surfix 20k 

    - For experiments on regularization on overlapping on Multi-MNIST: set NUM_IN_COMMON = [3], and run python multi_mnist.py --use-bounding-box-overlap --min-width-scale 0.6 --max-width-scale 0.6 --min-height-scale 0.6 --max-height-scale 0.6 --images-per-digit 20000 --name_surfix bbox20k 

    - For experiments on regularization on overlapping on Multi-dSprites: set NUM_IN_COMMON = [3], and run python multi_dsprites.py --use-bounding-box-overlap --min-width-scale 0.6 --max-width-scale 0.6 --min-height-scale 0.6 --max-height-scale 0.6 --images-per-digit 20000 --name_surfix bbox20k

## Regularization on number of objects:
    - Baseline (AIR) : python training_air_original.py -dn [num] -ds 20k -data [data] -gpu [gpu-id]. num is either 13, 14 or 24. data is either mnist or sprites.
    - Baseline (AIR-pPrior) : python train_air_pr.py -dn [num] -ds 20k -data [data] -gpu [gpu-id]. num is either 13, 14 or 24. data is either mnist or sprites.
    - ASR: python train_air_pr.py -dn [num] -ds 20k -gm 100. -gne 10. -data [data] -gpu [gpu-id]. num is either 13, 14 or 24. data is either mnist or sprites.

## Regularization on overlapping:
    - Baseline (AIR) : python training_air_original.py -dn 3 -ds bbox20k -ap true -data [data] -gpu [gpu-id]. data is either mnist or sprites.
    - Baseline (AIR-pPrior) : python train_air_pr.py -dn 3 -ds bbox20k -data [data] -gpu [gpu-id]. data is either mnist or sprites.
    - ASR: python train_air_pr.py -dn 3 -ds bbox20k -gb 1. -gs 10. -ga 20. -data [data] -gpu [gpu-id]. data is either mnist or sprites.

