#!/bin/bash
for ((n=0;n<300;n++))
do
echo running game $n
python train_boost_Galen.py -g $n -e '' -d '../../data_DR_L_split/agent_0/train/' > ../training_numpy_tempt_out_dir/temp-$n.out 2>&1 &
wait
echo finishing game $n
sleep 20s
done
exit 0
