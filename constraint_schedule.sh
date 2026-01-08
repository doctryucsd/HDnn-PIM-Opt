: "${CONSTRAINT:=false}"
: "${POLICY:=static}" # static, linear
: "${ACQF:=qExpectedHypervolumeImprovement}" # qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
: "${DATASET:=cifar10}"

if [ "$ACQF" = "qExpectedHypervolumeImprovement" ]; then
  METHOD=EHVI
elif [ "$ACQF" = "qNoisyExpectedHypervolumeImprovement" ]; then
  METHOD=NEHVI
fi

if [ "$CONSTRAINT" = true ]; then
  FILE_NAME_PREFIX=${DATASET}_${METHOD}_${POLICY}-constraint
else
  FILE_NAME_PREFIX=${DATASET}_${METHOD}_no-constraint
fi

# Configure runs here (not via CLI)
: "${RUNS:=5}"
: "${START:=144}"

mkdir -p logs

i=0
while [ "$i" -lt "$RUNS" ]; do
  SEED=$((START + i))
  FILE_NAME="${FILE_NAME_PREFIX}_seed${SEED}"
  cmd="python3 -m sim optimization.constrained=${CONSTRAINT} optimization.threshold_schedule=${POLICY} optimization.metrics_file=${FILE_NAME} seed=${SEED} optimization.acqf=${ACQF}"
  echo $cmd
  $cmd > "logs/${FILE_NAME}_seed${SEED}.log" 2>&1 &

  i=$((i + 1))
  if [ "$i" -lt "$RUNS" ]; then
    sleep 60
  fi
done

# Ensure all background runs finish before copying results
wait

# cp outputs/*/*/*.json /cimloop-volume/
