## Introduction

This project is based on AuctionGym by [Olivier Jeunen](https://github.com/amzn/auction-gym) with approaches in his [research paper](https://www.amazon.science/publications/learning-to-bid-with-auctiongym). Utilizing his work in simulating auctions, I am able to create different scenarios to test models' limits. The focus of this paper to to train tree based 
surroagte models to predict the optimal bid for a given auction. The models are trained based on a black box approach Doubly Robust Estimator. There is also a modified implementation
of Linear Model U Tree (LMUT) in src/c_utree_boost that allows the usage of LMUT to action space. The original implementation of LMUT comes from (https://github.com/Guiliang/uTree_mimic_mountain_car)

## Work

For the usage of LMUT, necessary pathes has to be recreated to storage models and training data. However, the training records are provided in training_numpy_tempt_out_dir/
There are also some changes in AuctionGym environment for creating a surrogate bidder to join the bidding process. The changes are in Bidder.py.

## Reproducing Research Results

Different secenarios are created based on their configure files in config/ . There are jupter notebooks to recreate experiments in my paper. They follow this naming pattern: 
 {estimator}_{competition}__{scenario}_new.ipynb. For example, DR_L_7_contexts.ipynb is the notebook to recreate the results for DR estimator in low competition mode with 7 contexts.
The only difference is the Gamma Prediction experiment and New Competition Added scenario. They are utilising the same data and config as DR_L.ipynb. So, the results are put in the same notebook.

Although the data is not included in this repo, the pickle file helps in recreating the data for surrogate model training, validation and testing .
```

## License

This project is licensed under the Apache-2.0 License.

