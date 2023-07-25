#!/bin/bash

t_paras=('DR')
c_paras=('L')
s_paras=(1000)
a_paras=(0 1 2)
p_paras=(10 20 35 50)
i_paras=(50)

MAX_PROCESSES=10

for ((n=28;n<44;n++))
do
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
            for a in "${a_paras[@]}"
            do
              echo running game $n with parameters -t $t -c $c -s $s -p $p -i $i -a $a
              (
                python train_boost_Galen.py -g $n -t $t -c $c -a $a -s $s -m 10000 -p $p -i $i -e '' -d "../../data_${t}_${c}_split_${s}_adaptiveTrain/agent_${a}/train/" > ../training_numpy_tempt_out_dir/${t}_${c}_adaptiveTrain/agent_${a}/split_${s}/max_hist_10000_max_depth_${p}_min_split_instances_${i}/temp-$n.out 2>&1
                echo finishing game $n
              ) &
                sleep 30s
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
