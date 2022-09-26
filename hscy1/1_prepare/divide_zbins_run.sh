#!/usr/bin/env bash
date
for ((j=92;j<94;j++)); do
    echo $j
    for ((i=$j*21;i<($j+1)*21;i++)); do
      srun --ntasks 1 --exclusive --mem-per-cpu 4G ./divide_zbins.py --id $i &
    done
    wait
done
wait
date
