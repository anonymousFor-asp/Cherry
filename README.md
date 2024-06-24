# Breaking the Memory Wall: An Efficient Micro-Batch Training Method for Graph Neural Network on Large Graphs

## Install Requirements

### Install with Docker

```shell
docker pull cherrywang/cherry:v1
```

## Install with virtual environment

Cherry is implemented using DGL 2.0.0 and PyTorch 2.0.1.

To be more specific, Cherry uses python 3.10, CUDA 11.8/cuDNN 8.7

```shell
bash ./requirements.sh # install cherry requirements
```

If you want to test the effect of Betty at the same time, please execute the following command additionally

```shell
bash ./pytorch/mirco_batch_train/Betty_file/gen_data.sh
```

For more details about Betty's experimental environment, please refer to[Betty Code](https://zenodo.org/records/7439846)

## File Organization

-  `pytorch/micro_batch_train` folder contains all the implementations of Cherry.

-  `cherry_graph_partitioner.py` realizes the Out-degree Centric Graph Partitioning

- `c_block_dataloader.py` is the old dataloader for vanilla cherry.

- `micro_batch_train.py` is Cherry's final version and contains MBLs construction and training.

- `pytorch/micro_batch_train/Betty_file` is Betty code from [Betty Code](https://zenodo.org/records/7439846), more details please read `pytorch/micro_batch_train/Betty_file/readme.md`

## Run Cherry on your device

1. `git clone https://github.com/anonymousFor-asp/Cherry.git`
2. Configure your environment according to the install requirements
3. Configure your dataset storage folder in `pytorch/micro_batch_train/utils/load_graph.py`
   - All datasets used in the experiment can be automatically downloaded except Amazon.You can manually download the amazon dataset from [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)

4- `./Evaluation` contains some test examples, including the writing of test scripts. You can adjust the parameters to test your own experiments.