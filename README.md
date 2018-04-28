# introduction
- attention based summarization on tensorflow using seq2seq model
- my graduation project code
- do not provide data for the time

# environment
- ubuntu 16.04 lts
- anaconda python 3.6
- recompiled tensorflow r1.7 gpu version
- CUDA 9.0
- cudnn 7.1.2
- rouge

# progress
- [x] finish word embedding matrix
- [x] build seq2seq model
- [x] test lstm and gru core
- [x] test bidirectional core
- [x] fix infer problem
- [x] test multilayer with dropout core
- [x] fix lazy loading
- [x] fix pre-processing
- [x] try training with non-mentor model
- [x] secondary activation
- [x] test attention decoder(luong attention)
- [x] choose last batch in each epoch as the validation set
- [x] learning rate decay:gradient descent,low init value,decay=0.995
- [x] cut vocab size to 1000,replace unusual word to unk
- [x] enlarge rnn hidden units size
- [x] fix word embedding matrix and try to load model
- [x] divide infer and train into two graphs
- [x] use rouge to value model
- [x] save each test result
- [ ] fix unk problems
- [x] train sentiment classification svm
- [ ] add sentiment-blended word embeddings
- [X] use larger corpus

# current effect
- test output can been seen under "./infer/output.txt"

# current best result
|         |            |         |                |         |   |          |
|:-------:|:----------:|:-------:|:--------------:|:-------:|:-:|:--------:|
| ROUGE-1 | Average_R: | 0.22519 | (95%-conf.int. | 0.16896 | - | 0.27646) |
| ROUGE-1 | Average_P: | 0.25361 | (95%-conf.int. | 0.19050 | - | 0.32250) |
| ROUGE-1 | Average_F: | 0.23207 | (95%-conf.int. | 0.17892 | - | 0.28701) |
|         |            |         |                |         |   |          |
| ROUGE-2 | Average_R: | 0.06118 | (95%-conf.int. | 0.02703 | - | 0.10253) |
| ROUGE-2 | Average_P: | 0.07478 | (95%-conf.int. | 0.03051 | - | 0.13185) |
| ROUGE-2 | Average_F: | 0.06286 | (95%-conf.int. | 0.02782 | - | 0.10325) |
|         |            |         |                |         |   |          |
| ROUGE-3 | Average_R: | 0.02120 | (95%-conf.int. | 0.00446 | - | 0.04353) |
| ROUGE-3 | Average_P: | 0.02719 | (95%-conf.int. | 0.00391 | - | 0.06641) |
| ROUGE-3 | Average_F: | 0.02060 | (95%-conf.int. | 0.00417 | - | 0.04179) |
|         |            |         |                |         |   |          |
| ROUGE-4 | Average_R: | 0.00000 | (95%-conf.int. | 0.00000 | - | 0.00000) |
| ROUGE-4 | Average_P: | 0.00000 | (95%-conf.int. | 0.00000 | - | 0.00000) |
| ROUGE-4 | Average_F: | 0.00000 | (95%-conf.int. | 0.00000 | - | 0.00000) |
|         |            |         |                |         |   |          |
| ROUGE-L | Average_R: | 0.20610 | (95%-conf.int. | 0.15658 | - | 0.25433) |
| ROUGE-L | Average_P: | 0.23725 | (95%-conf.int. | 0.17520 | - | 0.30571) |
| ROUGE-L | Average_F: | 0.21450 | (95%-conf.int. | 0.16301 | - | 0.26592) |