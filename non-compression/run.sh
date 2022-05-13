python3 main.py --model gcn --dataset MUTAG         > log/gcn_mutag.log
python3 main.py --model gcn --dataset PTC           > log/gcn_ptc.log
python3 main.py --model gcn --dataset IMDBMULTI --graph_pooling_type mean --neighbor_pooling_type sum --degree_as_nlabel  > log/gcn_imdbmulti.log

python3 main.py --model gin --dataset MUTAG         > log/gin_mutag.log
python3 main.py --model gin --dataset PTC           > log/gin_ptc.log
python3 main.py --model gin --dataset IMDBMULTI --graph_pooling_type mean --neighbor_pooling_type sum --degree_as_nlabel  > log/gin_imdbmulti.log