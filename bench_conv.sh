#!/bin/bash
function ProcessLog() {
  if grep -q ">>> Convolution Finished." $1; then
      echo "Looks Good!"
  else
      echo "Looks Bad!"
  fi
}

# Varying N
for (( i = 1 ; i <= 20 ; i++ )); do
  ./test_conv_bias_elu.out --input $i,8,10,10 --filter 8,8,2,2 --bias 1,8,1,1 &> log.tmp
  echo -n "N=$i C=8 K=8 HW=10 RS=2 "
  ProcessLog log.tmp
done

# Varying C
for (( i = 1 ; i <= 20 ; i++ )); do
  ./test_conv_bias_elu.out --input 3,$i,10,10 --filter 8,$i,2,2 --bias 1,8,1,1 &> log.tmp
  echo -n "N=3 C=$i K=8 HW=10 RS=2 "
  ProcessLog log.tmp
done

# Varying K
for (( i = 1 ; i <= 20 ; i++ )); do
  ./test_conv_bias_elu.out --input 3,8,10,10 --filter $i,8,2,2 --bias 1,$i,1,1 &> log.tmp
  echo -n "N=3 C=8 K=$i HW=10 RS=2 "
  ProcessLog log.tmp
done

# Varying HW
for (( i = 1 ; i <= 20 ; i++ )); do
  ./test_conv_bias_elu.out --input 3,8,$i,$i --filter 8,8,2,2 --bias 1,8,1,1 &> log.tmp
  echo -n "N=3 C=8 K=8 HW=$i RS=2 "
  ProcessLog log.tmp
done

# Varying RS
for (( i = 1 ; i <= 10 ; i++ )); do
  ./test_conv_bias_elu.out --input 3,8,10,10 --filter 8,8,$i,$i --bias 1,8,1,1 &> log.tmp
  echo -n "N=3 C=8 K=8 HW=10 RS=$i "
  ProcessLog log.tmp
done
