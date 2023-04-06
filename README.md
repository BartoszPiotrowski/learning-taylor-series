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

In order to train a GPT2-like transformer model, run:

```
mkdir data
cut -d'#' -f1 training_pairs > training_pairs.src
cut -d'#' -f2 training_pairs > training_pairs.tgt
python3 train.py \
    --train_data training_pairs \
    --lr 1e-5 \
    --batch_size 64 \
    --train_steps_max 100000 \
    --save_dir data \
```
