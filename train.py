import numpy as np
import tensorflow as tf
from model import TRANS_Model
from feature_engineering import make_dataset, make_dataset_duplicates, get_data_one

def train_and_evaluate():
    #vars
    num_prev_chunks = 5
    num_prev_periods = 5
    period = 15

    #read csvs and make datasets
    print("Creating Dataset:")
    data_chunks, targets_chunks, data_burstrate, targets_burstrate = make_dataset(num_prev_chunks=num_prev_chunks, num_prev_periods=num_prev_periods, period=period)
    '''order = np.arange(data_burstrate.shape[0])
    np.random.shuffle(order)
    print(order)
    data_burstrate = data_burstrate[order,:]
    targets_burstrate = targets_burstrate[order]'''
    #data_burstrate = data_burstrate/1000 #convert burstrates to kb/s
    #targets_burstrate = targets_burstrate/1000 #convert burstrates to kb/s
    print("")

    ####Chunks:

    '''#set up training data
    print("Setting up chunk data")
    print("")
    split_point_chunks = int(0.9*data_chunks.shape[0])

    train_x_chunks = data_chunks[0:split_point_chunks]
    train_y_chunks = targets_chunks[0:split_point_chunks]

    train_x_chunks = train_x_chunks.reshape((train_x_chunks.shape[0], train_x_chunks.shape[1], 1))
    train_y_chunks = train_y_chunks.reshape(-1,1)

    #initialize model
    print("Training chunk model")
    print("")
    trans_model_chunks = TRANS_Model(train_x_chunks.shape[1:])

    #train model
    trans_model_chunks.train(train_x_chunks, train_y_chunks)

    #set up test data
    test_x_chunks = data_chunks[split_point_chunks:]
    test_y_chunks = targets_chunks[split_point_chunks:]

    test_x_chunks = test_x_chunks.reshape((test_x_chunks.shape[0], test_x_chunks.shape[1], 1))
    test_y_chunks = test_y_chunks.reshape(-1,1)

    #evaluate
    eval_chunks = trans_model_chunks.evaluate(test_x_chunks, test_y_chunks)
    avg_chunks = np.average(targets_chunks[split_point_chunks:])
    
    #print(eval)
    print("")
    print("Chunk Sizes:")
    print("Mean Squared Error (bytes): ", eval_chunks[1])
    print("Mean Absolute Error (bytes): ", eval_chunks[0])
    print("Average Testing Chunk Size (bytes)", avg_chunks)
    print("")
    '''

    ####burst rates:
    
    #set up training data
    print("Setting up burst rate data")
    print("")
    split_point_burstrate = int(1*data_burstrate.shape[0])

    train_x_burstrate = data_burstrate[0:split_point_burstrate]
    train_y_burstrate = targets_burstrate[0:split_point_burstrate]

    train_x_burstrate = train_x_burstrate.reshape((train_x_burstrate.shape[0], train_x_burstrate.shape[1], 1))
    train_y_burstrate = train_y_burstrate.reshape(-1,1)

    #initialize model
    print("Training burst rate model")
    print("")
    trans_model_burstrate = TRANS_Model(train_x_burstrate.shape[1:])

    #train model
    trans_model_burstrate.train(train_x_burstrate, train_y_burstrate)

    #set up test data
    test_x_burstrate = data_burstrate[split_point_burstrate:]
    test_y_burstrate = targets_burstrate[split_point_burstrate:]

    test_x_burstrate = test_x_burstrate.reshape((test_x_burstrate.shape[0], test_x_burstrate.shape[1], 1))
    test_y_burstrate = test_y_burstrate.reshape(-1,1)

    #evaluate
    '''
    eval_burstrate = trans_model_burstrate.evaluate(test_x_burstrate, test_y_burstrate)
    avg_burstrate = np.average(targets_burstrate[split_point_burstrate:])
    preds = trans_model_burstrate.predict(test_x_burstrate)
    avg_percent_errors = tf.keras.metrics.mean_absolute_percentage_error(test_y_burstrate, preds)
    pct_under_100 = 0
    pct_over_100 = 0
    avg_under_100 = 0
    for i in range(avg_percent_errors.shape[0]):
        if(avg_percent_errors[i]<100):
            pct_under_100 += 1
            avg_under_100 += avg_percent_errors[i]
        else:
            pct_over_100 += 1
    
    avg_under_100 = float(avg_under_100)/float(pct_under_100)
    pct_over_100 = float(pct_over_100)/float(avg_percent_errors.shape[0])
    pct_under_100 = float(pct_under_100)/float(avg_percent_errors.shape[0])
    
    #print(eval)
    print("")
    print("Burst rates:")
    print("Mean Squared Error (bytes): ", eval_burstrate[1])
    print("Mean Absolute Error (bytes): ", eval_burstrate[0])
    print("Mean Absolute Percent Error (when under 100%): ", avg_under_100)
    print("Percent of predictions with <100 percent error: ", pct_under_100)
    print("Percent of predictions with >100 percent error: ", pct_over_100)
    print("Average Testing Burst rate (bytes)", avg_burstrate)
    print("")
    '''

    ####Testing on duplicate streams:

    #get duplicates data
    print("Getting Testing Data:")
    data_chunks_duplicates, targets_chunks_duplicates, data_burstrate_duplicates, targets_burstrate_duplicates = make_dataset_duplicates(num_prev_chunks=num_prev_chunks, num_prev_periods=num_prev_periods, period=period)
    print("")

    '''
    #evaluate duplicate chunks
    data_chunks_duplicates = data_chunks_duplicates.reshape((data_chunks_duplicates.shape[0], data_chunks_duplicates.shape[1], 1))
    targets_chunks_duplicates = targets_chunks_duplicates.reshape(-1,1)
    eval_chunks_duplicates = trans_model_chunks.evaluate(data_chunks_duplicates, targets_chunks_duplicates)
    avg_chunks_duplicates = np.average(targets_chunks)
    
    #print(eval_chunks)
    print("Chunk Sizes Duplicate:")
    print("Mean Squared Error (bytes): ", eval_chunks_duplicates[1])
    print("Mean Absolute Error (bytes): ", eval_chunks_duplicates[0])
    print("Average Chunk Size (bytes)", avg_chunks_duplicates)
    print("")
    '''

    
    #evaluate duplicate burst rates
    data_burstrate_duplicates = data_burstrate_duplicates.reshape((data_burstrate_duplicates.shape[0], data_burstrate_duplicates.shape[1], 1))
    targets_burstrate_duplicates = targets_burstrate_duplicates.reshape(-1,1)
    eval_burstrate_duplicates = trans_model_burstrate.evaluate(data_burstrate_duplicates, targets_burstrate_duplicates)
    avg_burstrate_duplicates = np.average(targets_burstrate_duplicates)
    preds = trans_model_burstrate.predict(data_burstrate_duplicates)
    avg_percent_errors_duplicates = tf.keras.metrics.mean_absolute_percentage_error(targets_burstrate_duplicates, preds)
    pct_under_100 = 0
    pct_over_100 = 0
    avg_under_100 = 0
    for i in range(avg_percent_errors_duplicates.shape[0]):
        if(avg_percent_errors_duplicates[i]<100):
            pct_under_100 += 1
            avg_under_100 += avg_percent_errors_duplicates[i]
        else:
            pct_over_100 += 1
    
    avg_under_100 = float(avg_under_100)/float(pct_under_100)
    pct_over_100 = float(pct_over_100)/float(avg_percent_errors_duplicates.shape[0])
    pct_under_100 = float(pct_under_100)/float(avg_percent_errors_duplicates.shape[0])
    
    #print(eval)
    print("Burst rates duplicates:")
    print("Mean Squared Error (bytes): ", eval_burstrate_duplicates[1])
    print("Mean Absolute Error (bytes): ", eval_burstrate_duplicates[0])
    print("Mean Absolute Percent Error (when under 100%): ", avg_under_100)
    print("Percent of predictions with <100 percent error: ", pct_under_100)
    print("Percent of predictions with >100 percent error: ", pct_over_100)
    print("Average Burst rate (bytes)", avg_burstrate_duplicates)
    

    #outs = trans_model.predict(test_x)
    #print("")
    #print(outs)

if __name__ == '__main__':
    train_and_evaluate()