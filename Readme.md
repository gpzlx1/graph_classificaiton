# Four dataset
* MUTAG
* PTC
* IMDBMULTI
* ogba-ppa

# Run

```shell
# for MUTAG
python3 main.py --model gcn --dataset MUTAG

# for PTC
python3 main.py --model gcn --dataset PTC

# for IMDBMULTI
python3 main.py --model gcn --dataset IMDBMULTI --graph_pooling_type mean --neighbor_pooling_type sum --degree_as_nlabel

# for ogba-ppa
python3 main_ogba_ppa.py
```