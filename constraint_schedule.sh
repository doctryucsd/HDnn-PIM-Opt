POLICY=linear # static, linear, exponential
FILE_NAME=cifar10_NEHVI_${POLICY}_random10

cmd="python3 -m sim optimization.threshold_schedule=${POLICY} optimization.metrics_file=${FILE_NAME}"
echo $cmd
$cmd > logs/${FILE_NAME}.log 2>&1