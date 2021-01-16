# DNA Sequencing using Machine learning
![Image](https://github.com/nageshsinghc4/DNA-Sequence-Machine-learning/blob/master/PX000098_PRESENTATION.jpeg)
The double-helix is the correct chemical representation of DNA. But DNA is special. It’s a nucleotide made of four types of nitrogen bases: Adenine (A), Thymine (T), Guanine (G) and Cytosine. We always call them A, C, Gand T.

A genome is a complete collection of DNA in an organism. All living species possess a genome, but they differ considerably in size.

As a data-driven science, genomics extensively utilizes machine learning to capture dependencies in data and infer new biological hypotheses. Nonetheless, the ability to extract new insights from the exponentially increasing volume of genomics data requires more powerful machine learning models. By efficiently leveraging large data sets, deep learning has reconstructed fields such as computer vision and natural language processing. It has become the method of preference for many genomics modeling tasks, including predicting the influence of genetic variation on gene regulatory mechanisms such as DNA receptiveness and splicing.

So here, we will understand DNA structure and how machine learning can be used to work with DNA sequence data.

Pre requisits:

1. **Biopython** :is a collection of python modules that provide functions to deal with DNA, RNA & protein sequence.

```pip install biopython```

2. **Squiggle** : a software tool that automatically generates interactive web-based two-dimensional graphical representations of raw DNA sequences.

```pip install Squiggle```

DNA sequence data usually are contained in a file format called “fasta” format. Fasta format is simply a single line prefixed by the greater than symbol that contains annotations and another line that contains the sequence:

***“AAGGTGAGTGAAATCTCAACACGAGTATGGTTCTGAGAGTAGCTCTGTAACTCTGAGG”***

In this repository, we are building a classification model that is trained on the human DNA sequence and can predict a gene family based on the DNA sequence of the coding sequence. To test the model, we will use the DNA sequence of humans, dogs, and chimpanzees and compare the accuracies.

You can read this article to understand the project step by step from [www.theaidream.com](https://www.theaidream.com/post/demystify-dna-sequencing-with-machine-learning-and-python) or my [kaggle notebook](https://www.kaggle.com/nageshsingh/demystify-dna-sequencing-with-machine-learning) for implementation.
