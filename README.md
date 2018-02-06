# introduction
attention based summarization on tensorflow using blstm seq2seq model

# environment
- ubuntu 16.04 lts
- anaconda python 3.6
- recompiled tensorflow r1.5 supporting avx and avx2
- pyrouge using rouge1.5.5

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

# usage
- still in developing
- it may make some extra dir like:graph(for tensorboard),log(for recording processing and loss),model(save the seq2seq mode),save(save the w2v embed matrix),processed(for pca evaluate on w2v)

# current effect
- group 1

     - infer headline: 
     report british economy poised for strong growth and possibly an interest rate hike hold for release until 103104 190100 est 
     - targets: 
     report british economy poised for strong growth and possibly an interest rate hike hold for release until 103104 190100 est 


- group 2

     - infer headline: 
     israel would not allow arafat to be buried at palestinian leadership 
     - targets: 
     israel would not allow arafat to be buried in jerusalem but affirms pledge to allow him to return to west bank if he recovers 


- group 3

     - infer headline: 
     and bank and gambling next year whether to hunt for anthrax 
     - targets: 
     leading union declares royal dutchshell enemy of the people in africas largest oil producer 


- group 4

     - infer headline: 
     government darfur rebels to be free to fight hivaids 
     - targets: 
     sudanese government says it accepts rebel political demands on darfur 


- group 5

     - infer headline: 
     un workers meet afghan president yearn for police on last australian army 
     - targets: 
     video shows kidnapped foreign election workers militants demand un withdrawal from afghanistan 


- group 6

     - infer headline: 
     ballot for tickets to lions matches opens 
     - targets: 
     ballot for tickets to lions matches opens 


- group 7

     - infer headline: 
     oil prices dips to close in october 
     - targets: 
     protesters hope to fend off retail store near ruins in mexico 


- group 8

     - infer headline: 
     missing 377 tons just a tiny fraction of munitions stashed across iraq 
     - targets: 
     missing 377 tons just a tiny fraction of munitions stashed across iraq 


- group 9

     - infer headline: 
     a look at some top leader in annual santa port 
     - targets: 
     a look at some top weapons sites in iraq 


- group 10

     - infer headline: 
     dollar higher against yen in tokyo 
     - targets: 
     dollar lower against yen in tokyo 
