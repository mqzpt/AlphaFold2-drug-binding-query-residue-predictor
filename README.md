# AlphaFold2-drug-binding-query-residue-predictor-neural-network
Implemented a neural network using Python, Pandas, NumPy, and Tensorflow, to predict 'drug binding' or 'non-drug binding' for any query residue on an AlphaFold2 predicted protein model. 

For those who wander:

A neural network is an approach to machine learning, inspired by neural networks in real human brains. The approach 

**How I am going to improve this model**

I am currently working on a XGB approach with a goal of at least 90% accuracy. This should fair more effectively than a basic neural network approach given the scale of the dataset, though we may need to worry about overfitting, as the dataset is a bit sparse and unstructed at times. I am working with numpy to find ways to clean the data as best as possible to try and avoid this issue.
