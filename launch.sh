#!/bin/bash
TIME=`date +\%s`
mkdir models/$TIME
sbatch --output=models/$TIME/log.out --export=TIME=$TIME job.slurm
