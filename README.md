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
- [ ] test attention decoder(luong attention)
- [x] choose last batch in each epoch as the validation set
- [x] learning rate decay:gradient descent,low init value,decay=0.995
- [x] cut vocab size to 1000,replace unusual word to unk
- [x] enlarge rnn hidden units size
- [x] fix word embedding matrix and try to load model
- [x] divide infer and train into two graphs
- [x] use rouge to value model
- [x] save each test result
- [ ] fix unk problems
- [ ] train sentiment classification svm
- [ ] add sentiment-blended word embeddings
- [ ] use large corpus

# current effect
- test output can been seen under "./infer/output.txt"