import numpy as np
import tensorflow as tf
from model import TRANS_Model
from feature_engineering import make_dataset, make_dataset_testing

def train_and_evaluate():
    #vars
    num_prev_periods = 5
    period = 15

    #read csvs and make datasets
    print("Creating Dataset:")
    data_burstrate, targets_burstrate = make_dataset(num_prev_periods=num_prev_periods, period=period)
    '''order = np.arange(data_burstrate.shape[0])
    np.random.shuffle(order)
    print(order)
    data_burstrate = data_burstrate[order,:]
    targets_burstrate = targets_burstrate[order]'''
    #data_burstrate = data_burstrate/1000 #convert burstrates to kb/s
    #targets_burstrate = targets_burstrate/1000 #convert burstrates to kb/s
    print("")

    
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


    ####Testing:

    #get testing data
    print("Getting Testing Data:")
    data_burstrate_testing, targets_burstrate_testing = make_dataset_testing(num_prev_periods=num_prev_periods, period=period)
    print("")

    
    #evaluate duplicate burst rates
    data_burstrate_testing = data_burstrate_testing.reshape((data_burstrate_testing.shape[0], data_burstrate_testing.shape[1], 1))
    targets_burstrate_testing = targets_burstrate_testing.reshape(-1,1)
    eval_burstrate_testing = trans_model_burstrate.evaluate(data_burstrate_testing, targets_burstrate_testing)
    avg_burstrate_testing = np.average(targets_burstrate_testing)
    preds = trans_model_burstrate.predict(data_burstrate_testing)
    avg_percent_errors_testing = tf.keras.metrics.mean_absolute_percentage_error(targets_burstrate_testing, preds)
    pct_under_100 = 0
    pct_over_100 = 0
    avg_under_100 = 0
    for i in range(avg_percent_errors_testing.shape[0]):
        if(avg_percent_errors_testing[i]<100):
            pct_under_100 += 1
            avg_under_100 += avg_percent_errors_testing[i]
        else:
            pct_over_100 += 1
    
    avg_under_100 = float(avg_under_100)/float(pct_under_100)
    pct_over_100 = float(pct_over_100)/float(avg_percent_errors_testing.shape[0])
    pct_under_100 = float(pct_under_100)/float(avg_percent_errors_testing.shape[0])
    
    #print(eval)
    print("Burst Rates Testing:")
    print("Mean Squared Error (bytes): ", eval_burstrate_testing[1])
    print("Mean Absolute Error (bytes): ", eval_burstrate_testing[0])
    print("Mean Absolute Percent Error (when under 100%): ", avg_under_100)
    print("Percent of predictions with <100 percent error: ", pct_under_100)
    print("Percent of predictions with >100 percent error: ", pct_over_100)
    print("Average Burst rate (bytes)", avg_burstrate_testing)
    

    #outs = trans_model.predict(test_x)
    #print("")
    #print(outs)

if __name__ == '__main__':
    train_and_evaluate()