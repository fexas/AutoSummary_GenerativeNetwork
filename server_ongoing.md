# Server

## HPC3

1. sir:
2. lotka volterra: dnn
3. toy example: dnnabc, bf

## Question Mark

1. for toy example nn, the reason for small acceptance ratio
   is
   1. small quantile level
      ```aiignore
      quantile_level=0.005 (determined)
      epsilon_upper_bound = 0.1
      ```
   2. small epsilon lower bound

## Record

```aiignore
 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1555595   gpu3090  sjob.bf    rlico  R       0:03      1 hhnode-ib-191
           1555594   gpu3090 sjob.nn_    rlico  R       2:38      1 hhnode-ib-190
           1555590   gpu3090 sjob.nn_    rlico  R       8:30      1 hhnode-ib-189
```

## What to do for tommorrow

slow down to speed up

1. toy example, improve the performance of NN and bayesflow
   1. (ps) Np to generate posterior is to large
   2. current epsilon threshold is too large, try 0.035
2. sbatch w2abc of toy
3. transfer toy example from 50 to 25 and 100
