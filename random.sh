: "${DATASET:=cifar10}"
FILE_NAME_PREFIX=${DATASET}_random

# Configure runs here (not via CLI)
: "${RUNS:=7}"
: "${START:=142}"

mkdir -p logs

i=0
while [ "$i" -lt "$RUNS" ]; do
  SEED=$((START + i))
  FILE_NAME="${FILE_NAME_PREFIX}_seed${SEED}"
  cmd="python3 -m sim optimization.metrics_file=${FILE_NAME} seed=${SEED} optimization.num_trials=10"
  echo $cmd
  $cmd > "logs/${FILE_NAME}_seed${SEED}.log" 2>&1 &

  i=$((i + 1))
  if [ "$i" -lt "$RUNS" ]; then
    sleep 60
  fi
done

# Ensure all background runs finish before copying results
wait

cp outputs/*/*/*.json /cimloop-volume/
