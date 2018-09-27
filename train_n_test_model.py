import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


df = pd.read_csv("nvda_stock_data.csv")

df=clean_dataset(df_orig)

X=df[['Year','Month','Day']].values

del df['Year']
del df['Month']
del df['Day']
del df['Share Price']
del df['P/E']

Y=df.drop('Share Price (End of Period)', axis=1).values
X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size=0.3, random_state=0)

# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# Define model parameters
RUN_NAME = "run 1 with 50 nodes"
learning_rate = 0.001
training_epochs = 150
display_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs = 3
number_of_outputs = 29

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 150
layer_2_nodes = 250
layer_3_nodes = 150

# Section One: Define the layers of the neural network itself
# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32,shape=(None,29))
    cost = tf.reduce_mean(tf.squared_difference(prediction,Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Create a summary operation to log the progress of the network
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

saver = tf.train.Saver()
# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:
    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):
        learning_rate = tf.train.exponential_decay(0.15, epoch, 1, 0.9999)

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        # Create log file writers to record training progress.
        # We'll store training and testing log data separately.
        training_writer = tf.summary.FileWriter("./logs3_2/{}/training".format(RUN_NAME), session.graph)
        testing_writer = tf.summary.FileWriter("./logs3_2/{}/testing".format(RUN_NAME), session.graph)

        # Every 5 training steps, log our progress
        if epoch % 5 == 0:
            training_cost, training_summary = session.run([cost,summary], feed_dict={X: X_scaled_training, Y:Y_scaled_training})
            prediction_val, testing_cost, testing_summary = session.run([prediction,cost,summary], feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})

            # Write the current training status to the log files (Which we can view with TensorBoard)
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

            print(epoch, training_cost, testing_cost)

    # Training is now complete!
    print("Training is complete!")

    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))

