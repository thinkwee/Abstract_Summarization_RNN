# introduction
- attention based summarization on tensorflow using seq2seq model
- my graduation project code
- do not provide data for the time

# environment
- ubuntu 16.04 lts
- anaconda python 3.6
- recompiled tensorflow r1.5 supporting avx and avx2
- ~~pyrouge using rouge 1.5.5~~

# refering
- Rush A M, Chopra S, Weston J. A Neural Attention Model for Abstractive Sentence Summarization[J]. Computer Science, 2015.
- Nallapati R, Zhou B, Santos C N D, et al. Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond[J].  2016.
- Luong M T, Pham H, Manning C D. Effective Approaches to Attention-based Neural Machine Translation[J]. Computer Science, 2015.
- Bahdanau D, Cho K, Bengio Y. Neural Machine Translation by Jointly Learning to Align and Translate[J]. Computer Science, 2014.
- Pang T B, Pang B, Lee L. Thumbs up? Sentiment Classification using Machine Learning[J]. Proceedings of Emnlp, 2002:79-86.
- Serban I V, Sordoni A, Bengio Y, et al. Building end-to-end dialogue systems using generative hierarchical neural network models[J].  2015（4）:3776-3783.
- He Y, Su W, Tian Y, et al. Summarizing Microblogs on Network Hot Topics[C]// International Conference on Internet Technology and Applications. IEEE, 2011:1-4.
- Lin P, Xiao R, Zhang Y. News event summarization complemented by micropoints[C]// IEEE International Conference on Data Engineering Workshops. IEEE, 2015:190-197.
- Shen, Shiqi, Zhao, Yu, Liu, Zhiyuan, Sun, Maosong,et al. Neural headline generation with sentence-wise optimization. arXiv preprint arXiv:1604.01904, 2016.
- Sutskever, I., Vinyals, O., and Le, Q. （2014）. Sequence to sequence learning with neural networks.In Advances in Neural Information Processing Systems （NIPS 2014）.
- Shen T, Zhou T, Long G, et al. DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding[J]. 2017.
- Suzuki J, Nagata M. Cutting-off Redundant Repeating Generations for Neural Abstractive Summarization[C]// Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers. 2017:291-297.
- Gehring J, Auli M, Grangier D, et al. Convolutional Sequence to Sequence Learning[J]. 2017.

# progress
- [x] finish word embedding matrix
- [x] build seq2seq model
- [x] test lstm and gru core
- [x] test bidirectional core
- [x] fix infer problem
- [x] test multilayer with dropout core
- [ ] test attention decoder(luong attention)
- [x] choose last batch in each epoch as the validation set
- [x] learning rate decay:gradient descent,low init value,decay=0.995
- [x] cut vocab size to 1000,replace unusual word to unk
- [x] enlarge rnn hidden units size
- [x] fix word embedding matrix and try to load model
- [x] divide infer and train into two graphs
- [ ] use rouge to value model
- [ ] save each test result
- [ ] fix unk problems
- [ ] train sentiment classification svm
- [ ] add sentiment-blended word embeddings
- [ ] use large corpus

# temp dir
- graph(for tensorboard)
- log(for recording processing and loss)
- model(save the seq2seq mode)
- save(save the w2v embed matrix)
- processed(for pca evaluate on w2v)

# current effect
- test output can been seen under "./infer/output.txt"