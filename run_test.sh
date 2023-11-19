#!/bin/bash
set -e
# modelArray = ("AFCN" "SFCN")
attackArray=("ori" "FGSM" "PGD" "GN")
test_attackArray=("PGD" "FGSM")
attackmodeArray=("none") # "bptt" "bptr")
# eval "nohup python -u main_test.py --model AFCN --identifier 'AFCN_FGSM[0.005000][ ]_l2[0.000500]' --name FGSM_PGD --attack PGD --eps -1 >tlogs/ANN-FGSM0.005-PGD.log>&1 &"
# for attack in "${attackArray[@]}"
# do
#     if [ "${attack}" = "ori" ]; then
#         identifier="'AFCN_clean_l2[0.000500]'"
#     else
#         identifier="'AFCN_${attack}[0.005000][ ]_l2[0.000500]'"
#     fi
#     for test_attack in "${attackArray[@]}"
#     do
#         test="nohup python -u main_test.py --model AFCN --identifier ${identifier} --name ${attack}_${test_attack} --attack ${test_attack} --eps -1 >tlogs/ANN-${attack}0.005-${test_attack}.log>&1 &"
#         echo "拼接结果：$test"
#         eval "${test}"
#         # 等待一分钟
#         sleep 20
#         # 继续执行其他命令
#         echo "等待结束，继续执行其他命令"
#     done
# done

for attack in "${attackArray[@]}"
do
    for attackmode in "${attackmodeArray[@]}"
    do
        if [ "${attack}" = "ori" ]; then
            identifier="'SFCN_clean_l2[0.000500]'"
        else
            identifier="'SFCN_${attack}[0.005000][${attackmode}]_l2[0.000500]'"
        fi
        for test_attack in "${test_attackArray[@]}"
        do
            test="nohup python -u main_test.py --model SFCN --identifier ${identifier} --name ${attack}_${test_attack} -T 8 --attack ${test_attack} --eps -1 --attack_mode ${attackmode}>tlogs/SNN-${attack}0.005-${test_attack}-${attackmode}.log>&1 &"
            echo "拼接结果：$test"
            eval "${test}"
            # 等待一分钟
            sleep 40
            # 继续执行其他命令
            echo "等待结束，继续执行其他命令"
        done
    done
done
# eval "nohup python -u main_test.py --model SFCN --identifier 'SFCN_clean_l2[0.000500]' --name ori_GN --attack GN -T 8 --eps -1 --attack_mode bptt >test_logs/SNN-ori-GN-bptt.log>&1 &"
# # 等待一分钟
# sleep 40
# # 继续执行其他命令
# echo "等待结束，继续执行其他命令"
# eval "nohup python -u main_test.py --model SFCN --identifier 'SFCN_clean_l2[0.000500]' --name ori_GN --attack GN -T 8 --eps -1 --attack_mode bptr >test_logs/SNN-ori-GN-bptr.log>&1 &"
# # 等待一分钟
# sleep 40
# # 继续执行其他命令
# echo "等待结束，继续执行其他命令"
# eval "nohup python -u main_test.py --model SFCN --identifier 'SFCN_FGSM[0.005000][bptr]_l2[0.000500]' --name FGSM_PGD --attack PGD -T 8 --eps -1 --attack_mode bptr >test_logs/SNN-FGSM-PGD-bptr.log>&1 &"
# 等待一分钟
# sleep 40
# # 继续执行其他命令
# echo "等待结束，继续执行其他命令"
