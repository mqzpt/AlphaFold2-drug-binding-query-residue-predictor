# AlphaFold2-drug-binding-query-residue-predictor-neural-network
Implemented a neural network using Python, Pandas, NumPy, and Tensorflow, to predict 'drug binding' or 'non-drug binding' for any query residue on an AlphaFold2 predicted protein model. 

*This was made for a data science challenge held by **Cyclica**, an AI drug discovery based in my hometown Toronto*

**For those who wander:**

A neural network is an approach to machine learning, inspired by how neurons work in real human brains. The approach works using layers of nodes between an input layer and an output layer, which when fed certain weights and biases can "learn", essentially activating certain nodes and passing along select data until reaching that final layer of neurons, where a decision can be made about the data.

**How I am going to improve this model:**

I am currently working on a XGB approach with a goal of at least 90% accuracy. This should fair more effectively than a basic neural network approach given the scale of the dataset, though we may need to worry about overfitting, since the dataset is a bit sparse and unstructed at times. I am working with numpy to find ways to clean the data as best as possible, as well as trying to gain a better understanding of the chemistry behind the data for improved feature selection to try and avoid this issue. This approach will also require a deeper level of mathematics and critical thinking as I won't be relying on Tensorflow to take care of algo.

**Help me!**

To test this model, make sure your TF environment is configured correctly. For those using an M1/M2 mac, you can optimize performance by taking advantage of the GPU with TF Metal. Here is a great tutorial I found for the env setup: https://www.youtube.com/watch?v=2C-B1VFMq58

If you run into any issues or have any questions or advice about this project, feel free to message me! Contact info on my profile.
