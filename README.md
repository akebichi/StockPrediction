# StockPrediction
A ML approach to predicting stocks based on financial metrics

In growit.py I show an incremental way of building the neural network. Eventhough TensorFlow does not allow for dynamic tree modification, I am using a mask that allow hiding weights and exposing them when needed. The resulting networks are a lot smaller than those used in static method. Will be adding empirical data later.
kaggle_inc.py shows how to add a layer to a pre-trained CNN. kaggle is an image dataset and is used here for showing a way to increase the size of a CNN, as a proof that incremental construction could be applied to more than just fully connected layers.
Another application of incremental architecture using LSTMs will be added later.

