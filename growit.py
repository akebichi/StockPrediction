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


df_orig = pd.read_csv("nvda_stock_data.csv")

#df_orig = pd.read_csv("all_companies_2.csv")
df=clean_dataset(df_orig)
del df['Share Price']
del df['P/E']

X=df.drop('Share Price (End of Period)', axis=1).values
Y=df[['Share Price (End of Period)']].values

X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size=0.3, random_state=0)

X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)


# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)


# Define model parameters
RUN_NAME = "run 1"
learning_rate = 0.001
training_epochs = 20 ##150
display_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs = 32 #33
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 100
layer_2_nodes = 200
layer_3_nodes = 100

nbr_limit=5

# Section One: Define the layers of the neural network itself

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
training_cost=1.0
prev_training_cost = 1.0
cnt=-1
cnt2=-1
cnt3=-1

print ("start training...")
while training_cost > 0.01:
    cnt=cnt+1
    print ("New growit iteration>>>>> ")

# Layer 1
    with tf.variable_scope('layer_1'):
        _mask = np.zeros(layer_1_nodes)
        print ("UnMasking ",cnt+1," neurons in layer1")
        for i in range(max(100,cnt)):
            _mask[i] = 1
        if cnt==0:
            foo = tf.get_variable(name="weights1_pre", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
            weights = tf.get_variable(name="weights1", initializer=foo )
            biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer() )
        layer_1_output = tf.nn.relu(tf.matmul(X, weights*_mask) + biases)

# Layer 2

    with tf.variable_scope('layer_2'):
        _mask = np.ones(layer_2_nodes)
        if cnt > nbr_limit:
            _mask = np.zeros(layer_2_nodes)
            cnt2=cnt2+1
            print ("UnMasking ",cnt2+1," neurons in layer2")
            for i in range(max(100, cnt2)):
                _mask[i] = 1

        if cnt==0:
            weights2 = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
            biases2 = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
        if cnt > nbr_limit:
            layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights2*_mask) + biases2)
        else:
            layer_2_output = tf.matmul(layer_1_output, weights2*_mask)  #identity fct


# Layer 3
    with tf.variable_scope('layer_3'):
        _mask = np.ones(layer_3_nodes)
        if cnt2 > nbr_limit:
            _mask = np.zeros(layer_3_nodes)
            cnt3=cnt3+1
            print ("UnMasking ",cnt3+1," neurons in layer3")
            for i in range(max(100, cnt3)):
                _mask[i] = 1
        if cnt==0:
            weights3 = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
            biases3 = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())

        if cnt2 > nbr_limit:
            layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights3*_mask) + biases3)
        else:
            layer_3_output = tf.matmul(layer_2_output, weights3*_mask)


# Output Layer
    with tf.variable_scope('output'):
        _mask = np.ones(number_of_outputs)

        if cnt==0:
            weights4 = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
            biases4 = tf.get_variable(name="biases3", shape=[number_of_outputs], initializer=tf.zeros_initializer())
        prediction = tf.nn.relu(tf.matmul(layer_3_output, weights4*_mask) + biases4)


# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training

    if cnt==0:
        with tf.variable_scope('cost'):
            Y = tf.placeholder(tf.float32,shape=(None,1))
            cost = tf.reduce_mean(tf.squared_difference(prediction,Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network

    if cnt == 0:
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Create a summary operation to log the progress of the network
    if cnt==0:
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
            session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training}) #1st arg was optimizer

        # Create log file writers to record training progress.
        # We'll store training and testing log data separately.
            training_writer = tf.summary.FileWriter("./logs2_2/{}/training".format(RUN_NAME), session.graph)
            testing_writer = tf.summary.FileWriter("./logs2_2/{}/testing".format(RUN_NAME), session.graph)


        # Every 5 training steps, log our progress
            if epoch % 5 == 0:
                prev_training_cost=training_cost
                training_cost, training_summary = session.run([cost,summary], feed_dict={X: X_scaled_training, Y:Y_scaled_training})
                prediction_val, testing_cost, testing_summary = session.run([prediction,cost,summary], feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})

            # Write the current training status to the log files (Which we can view with TensorBoard)
                training_writer.add_summary(training_summary, epoch)
                testing_writer.add_summary(testing_summary, epoch)

                print(epoch, training_cost, testing_cost)

print("Training is complete!")
