#!/bin/bash
function ProcessLog() {
  if grep -q ">>> MatMul Finished." $1; then
      echo "Looks Good!"
  else
      echo "Looks Bad!"
  fi
}

# Varying M
for (( i = 1 ; i <= 20 ; i++ )); do
  ./test_matmul_bias_activation.out --input0 1,$i,8 --input1 1,8,8 --bias 1,1,8 &> log.tmp
  echo -n "M=$i K=8 N=8 "
  ProcessLog log.tmp
done

# Varying K
for (( i = 1 ; i <= 20 ; i++ )); do
  ./test_matmul_bias_activation.out --input0 1,8,$i --input1 1,$i,8 --bias 1,1,8 &> log.tmp
  echo -n "M=8 K=$i N=8 "
  ProcessLog log.tmp
done

# Varying N
for (( i = 1 ; i <= 20 ; i++ )); do
  ./test_matmul_bias_activation.out --input0 1,8,8 --input1 1,8,$i --bias 1,1,$i &> log.tmp
  echo -n "M=8 K=8 N=$i "
  ProcessLog log.tmp
done
