#!/bin/bash
set -e
attackArray=("PGD")

for attack in "${attackArray[@]}"
do   
    train="nohup python -u main_train.py --name ${attack} -T 8 --attack ${attack} --eps 0.005 -atk_m none >train_logs/${attack}_train.log>&1 &"
    echo "拼接结果：$train"
    eval "${train}"
    # 等待一分钟
    sleep 40
    # 继续执行其他命令
    echo "等待结束，继续执行其他命令"
done