POLICY=low # static, low, high
FILE_NAME_PREFIX=cifar10_NEHVI_${POLICY}-constraint

if [ "$POLICY" = "static" ]; then
    ACCURACY=0.3
    POWER=0.3
    PERFORMANCE=0.3
    AREA=0.3
elif [ "$POLICY" = "low" ]; then
    ACCURACY=0.25
    POWER=0.35
    PERFORMANCE=0.35
    AREA=0.35
elif [ "$POLICY" = "high" ]; then
    ACCURACY=0.35
    POWER=0.25
    PERFORMANCE=0.25
    AREA=0.25
else
    echo "Invalid POLICY: $POLICY"
    exit 1
fi

# Configure runs here (not via CLI)
RUNS=5
START=142

mkdir -p logs

i=0
while [ "$i" -lt "$RUNS" ]; do
  SEED=$((START + i))
  FILE_NAME="${FILE_NAME_PREFIX}_seed${SEED}"
  cmd="python3 -m sim optimization.constrained=true optimization.threshold_schedule=static optimization.metrics_file=${FILE_NAME} seed=${SEED} optimization.constraints.accuracy=${ACCURACY} optimization.constraints.power=${POWER} optimization.constraints.performance=${PERFORMANCE} optimization.constraints.area=${AREA}"
  echo $cmd
  $cmd > "logs/${FILE_NAME}_seed${SEED}.log" 2>&1 &

  i=$((i + 1))
  if [ "$i" -lt "$RUNS" ]; then
    sleep 60
  fi
done

wait
