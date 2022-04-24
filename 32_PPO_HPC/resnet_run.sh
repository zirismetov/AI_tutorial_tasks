#!/bin/sh -v
#PBS -e /mnt/home/abstrac01/logs
#PBS -o /mnt/home/abstrac01/logs
#PBS -q batch
#PBS -l nodes=1:ppn=8:gpus=1:shared,feature=v100
#PBS -l mem=40gb
#PBS -l walltime=96:00:00
#PBS -A ditf_ldi
#PBS -N zafar_iris

eval "$(conda shell.bash hook)"
conda activate conda_env
export LD_LIBRARY_PATH=/mnt/home/abstrac01/.conda/envs/conda_env/lib:$LD_LIBRARY_PATH

cd /mnt/home/abstrac01/zafar_iris
python resnet_main.py -learning_rate 1e-3 -run_name testing
wait