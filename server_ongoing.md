# Server

## HPC3


1. toy example 100:w2
2. toy example: w2 replicate
3. lotka: dnn

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
           1562088   gpu3090 sjob.w2_lv    rlico  R       0:45      1 hhnode-ib-188
           1562086   gpu3090 sjob.w25    rlico  R       5:46      1 hhnode-ib-187
           1562084   gpu3090 sjob.w10    rlico  R      12:13      1 hhn
```

## What to do for tommorrow

slow down to speed up

1. toy example, improve the performance of NN and bayesflow
   1. (ps) Np to generate posterior is to large
   2. current epsilon threshold is too large, try 0.035
2. sbatch w2abc of toy
3. transfer toy example from 50 to 25 and 100

## Issue

1. shift in SIR
2. accept rate in w2abc and dnnabc
3. quan1 setting in toy
