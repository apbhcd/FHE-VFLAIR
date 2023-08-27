#!/bin/bash

for i in {1..20}
do
    python3 ./main_tree.py --seed 0 --number_of_trees $i --configs basic_configs_tree
    echo $i
done

