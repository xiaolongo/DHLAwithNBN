dataset=('DD')
model='SAGEDHLA'
norm='neighbornorm'
num_layers=(2 4 8 16 32 64)

cat /dev/null >results/graph_results.log
for d in ${dataset[@]}; do
    echo 'Dataset:' $d >>results/graph_results.log
    for i in ${num_layers[@]}; do
        echo 'Number of Layer:' $i >>results/graph_results.log
        /usr/bin/python run_graph.py \
            --dataset $d \
            --model $model \
            --norm $norm \
            --num_layers $i >>results/graph_results.log
    done
done
