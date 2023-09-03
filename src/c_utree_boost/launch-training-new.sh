#!/bin/bash
# sometimes the program is creating too fast that the function in Agent is trying to create path at the same time
# run this program will solve this problem
t_paras=('DR')
c_paras=('L')
s_paras=(1000)
a_paras=(0 1 2)
p_paras=(10 20 35)
i_paras=(50)
o_paras=(5)
special=''

MAX_PROCESSES=10

function run_game {
  t=$1
  c=$2
  s=$3
  p=$4
  i=$5
  o=$6
  a=$7
  for ((n=20;n<60;n++))
  do
    echo running game $n with parameters -t $t -c $c -s $s -p $p -i $i -o $o -a $a
    python train_boost_Galen.py -g $n -t $t -c $c -o $o -a $a -s $s -m 10000 -p $p -i $i -e '_linear_epoch_decay_lr' -d "../../data_${t}_${c}_split_${s}${special}/agent_${a}/train/" > ../training_numpy_tempt_out_dir/${t}_${c}${special}/agent_${a}/split_${s}/max_hist_10000_max_depth_${p}_min_split_instances_${i}_linear_epoch_decay_lr/temp-$n.out 2>&1
    echo finishing game $n
    sleep 20s
  done
}

for t in "${t_paras[@]}"
do
  for c in "${c_paras[@]}"
  do
    for s in "${s_paras[@]}"
    do
      for p in "${p_paras[@]}"
      do
        for i in "${i_paras[@]}"
        do
          for o in "${o_paras[@]}"
          do
            for a in "${a_paras[@]}"
            do
              run_game $t $c $s $p $i $o $a &
              while [ $(jobs -r | wc -l) -ge $MAX_PROCESSES ]; do
                sleep 1
              done
            done
          done
        done
      done
    done
  done
done

wait

exit 0