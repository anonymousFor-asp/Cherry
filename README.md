## File Organization

The `pytorch/micro_batch_train` folder contains all the implementations of Cherry.
Among them, `cherry_graph_partitioner.py` realizes the Out-degree Centric Graph Partitioning, `c_block_dataloader.py` is the old dataloader for vanilla cherry.
`micro_batch_train.py` is Cherry's final version and contains MBLs construction and training.

`pytorch/micro_batch_train/Betty_file` is Betty code from https://zenodo.org/records/7439846, more details please read `pytorch/micro_batch_train/Betty_file/readme.md`

To run Cherry on your platform, please configure all packages in `./requirements.sh` first
Then, You need to configure your dataset storage folder in `pytorch/micro_batch_train/utils/load_graph.py`

All datasets used in the experiment can be automatically downloaded except Amazon.
You can manually download the amazon dataset from [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)

`./Evaluation` contains some test examples, including the writing of test scripts. You can adjust the parameters to test your own experiments.

## Hardware requirements

The CPU needs at least 8 cores

* full evaluation
  GPU Memory >= 40GB
  Host Memory >= 200GB

* easy evaluation(just test reddit, ogbn-arxiv, ogbn-products)
  GPU Memory >= 24GB
  Host Memory >= 64GB