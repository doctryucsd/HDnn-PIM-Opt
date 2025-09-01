CONSTRAINT=true
POLICY=exponential # static, linear, exponential

if [ "$CONSTRAINT" = true ]; then
  FILE_NAME_PREFIX=cifar10_NEHVI_${POLICY}-constraint
else
  FILE_NAME_PREFIX=cifar10_NEHVI_no-constraint
fi

# Configure runs here (not via CLI)
RUNS=10
START=42

mkdir -p logs

i=0
while [ "$i" -lt "$RUNS" ]; do
  SEED=$((START + i))
  FILE_NAME="${FILE_NAME_PREFIX}_seed${SEED}"
  cmd="python3 -m sim optimization.constrained=${CONSTRAINT} optimization.threshold_schedule=${POLICY} optimization.metrics_file=${FILE_NAME} --seed=${SEED}"
  echo $cmd
  $cmd > "logs/${FILE_NAME}_seed${SEED}.log" 2>&1 &

  i=$((i + 1))
  if [ "$i" -lt "$RUNS" ]; then
    sleep 60
  fi
done

wait
