dataset='PPI'

num_layers=(2 4 8 16 32 64)
gpu_id=0
patience=100
norm=('neighbornorm')
mode=('max' 'mean' 'sum' 'concat')

cat /dev/null >results/ppi_results.log

model='GAT_with_JK'
echo 'Running Model:' $model >>results/ppi_results.log
for n in ${norm[@]}; do
    echo 'Normalization:' $n >>results/ppi_results.log
    for m in ${mode[@]}; do
        echo 'Mode:' $m >>results/ppi_results.log
        for i in ${num_layers[@]}; do
            echo 'Number of Layer:' $i >>results/ppi_results.log
            /usr/bin/python run_ppi.py \
                --model $model \
                --mode $m \
                --dataset $dataset \
                --num_layers $i \
                --patience $patience \
                --norm $n \
                --logger \
                --gpu_id $gpu_id >>results/ppi_results.log
        done
    done
done
