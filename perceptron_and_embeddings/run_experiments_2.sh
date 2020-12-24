for lr in 0.005 0.01 0.025 0.05 0.1
do
  for e in 1 2 3 4 5 10 15 25 50 100
  do 
    echo "$lr $e "
     python3 perceptron.py data/train_small.dat data/test_small.dat $lr $e
     echo ""
  done
done
