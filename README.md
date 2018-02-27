# introduction
attention based summarization on tensorflow using seq2seq model

# environment
- ubuntu 16.04 lts
- anaconda python 3.6
- recompiled tensorflow r1.5 supporting avx and avx2
- pyrouge using rouge 1.5.5

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
- try cross validation
- add bgru seq2seq model
- add switch module of core and test data
- add multilayer encoder
- add dropout
- add attention decoder(luong attention)

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

# problems to solve
- bad ability of generation
- can not calculate the loss of infer
- can not continue training from checkpoint
- can be improved by embed matrix pretrained on large corpus
- divide data in train,validate and test
- bad code structure
