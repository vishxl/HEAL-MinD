# HEAL-MinD

## Description
This repository contains the implementation of a novel model called HEAL-MinD for mental health detection over online customer's posts related to data breach, sensitive data loss, misuse of personal data etc. This model is based on deep learning which consists of several layers. It commences with an input layer which executes the input text and passed to the embedding layer. Two embeddings (BERT and GloVe) are taken in the embedding layer which receive the input text and their separate outcomes are combined together and that gives hybrid embedding. Thereafter, CGAT layer is taken which receives the hybrid embedding as input which learns the sequential and contextual representations in both directions. The CGAT consists of a CNN, forward GRU and backward GRU and their corresponding attention layers which give a hybrid context vector after combining it. Moreover, rich set of hand-crafted linguistics features along with a newly introduce mental health lexicon are taken to make the model well-informed. As a results, it gives a linguistics vector. Thereafter, a fused vector is generated after combining a hybrid context vector and linguistics vector. Finally, dense and output layers prepare the model for final classification and classify the input text as either mental health or non-mental health. The empirical evaluation of the proposed model is performed over three datasets and it shows a remarkably better performance as compared to the existing studies and several baseline methods.

## Requirements

### Dependencies
```- Python 3.8.11
- Keras 2.10.0
- NumPy 1.24.0
- Transformers 4.46.3
- TensorFlow 2.10.0
- Pandas 1.5.1
```
### Pre-trained Models
- GloVe 6B 300d
- BERT

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Citation
[Citation details to be added]

## Contact
[Contact information to be added]
