# introduction
attention based summarization on tensorflow using seq2seq model

# environment
- ubuntu 16.04 lts
- anaconda python 3.6
- recompiled tensorflow r1.5 supporting avx and avx2
- ~~pyrouge using rouge 1.5.5~~

# refering
- Neural Machine Translation By Jointly Learning To Align And Translate
- Effective Approaches to Attention-based Neural Machine Translation
- A Neural Attention Model for Abstractive Sentence Summarization	
- Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond	

# progress
- finish word embedding matrix
- add basic_rnn seq2seq model
- add test unit
- add save module
- add lstm seq2seq model
- add blstm seq2seq model
- fix infer problem
- add bgru seq2seq model
- add switch module of core and test data
- add multilayer encoder
- add dropout for multilayer lstm
- add attention decoder(luong attention)
- add learning rate dacay

# usage
- still in developing
- it may make some extra dir like
    - graph(for tensorboard)
    - log(for recording processing and loss)
    - model(save the seq2seq mode)
    - save(save the w2v embed matrix)
    - processed(for pca evaluate on w2v)

# current effect
- test output can been seen under "./infer/output.txt"

# to do
- [ ] choose one batch randomly as the validation set in each epoch
- [ ] learning rate decay:gradient descent,low init value,decay=0.995
- [ ] cut vocab size to 1000,replace unusual word to unk
- [ ] enlarge rnn hidden units size
- [ ] fix word embedding matrix and try to load model
- [ ] divide infer and train into two graphs
- [ ] use rouge to value model
- [ ] save each test result
- [ ] fix unk problems
- [ ] train sentiment classification svm
- [ ] add sentiment-blended word embeddings
- [ ] enlarge corpus
