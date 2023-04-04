# learning-taylor-series
Learning Taylor expension of functions with LSTM and Transformers.

In order to generate `N` training pairs of the form
```
(F, Taylor expansion of F up the fourth order)
```

run
```
./generate.py $N > training_pairs
```

In order to train a simple LSTM encoder-decoder model on the trainig data, run
```
./train-lstm.py training_pairs
```

