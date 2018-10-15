# StockPrediction
A ML approach to predicting stocks based on financial metrics

In growit.py I show an incremental way of building the neural network. Eventhough TensorFlow does not allow for dynamic tree modification, I am using a mask that allow hiding weights and exposing them when needed. The resulting networks are a lot smaller than those used in static method. Will be adding empirical data later.
