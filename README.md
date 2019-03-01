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

# run
- This work use Gigaword dataset which is not for public. You need fetch the data yourself.
- The SentiWordNet 3.0 dataset can be found here :[SentiWordNet3.0](https://drive.google.com/open?id=0B0ChLbwT19XcOVZFdm5wNXA5ODg)
- The codes are written in an early version of tensorflow. I do not recommend run this code directly. Just for reference.
- run ```python main.py -help``` for help.
- run ```python main.py -w2v``` to train the wordvector from Gigaword dataset using Word2Vec，then run ```python main.py -train``` to train the model and ```python main.py -test```to test the model(just get the output of testset).
- you need install ROUGE to test the output. All the results are collected in the original PERL version of ROUGE. Using PyRouge make cause the result a little bit higher.

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
- [x] cut vocab size to 3000,replace unusual word to unk
- [x] enlarge rnn hidden units size
- [x] fix word embedding matrix and try to load model
- [x] divide infer and train into two graphs
- [x] use rouge to value model
- [x] save each test result
- [ ] ~~fix unk problems~~
- [x] train sentiment classification svm
- [x] add sentiment-blended word embeddings
- [x] test sentiment classify
- [X] use larger corpus
- [x] collect ROUGE

# current effect
- ROUGE files collected in the './ROUGE_ANSWER'
