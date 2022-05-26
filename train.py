import numpy as np
from model import TRANS_Model
from feature_engineering import make_dataset, get_data_one

def train_and_evaluate():
    #number of previous chunks
    num_prev_chunks = 5

    #read csvs and make chunk size dataset
    data, targets = make_dataset(num_prev_chunks)

    #set up training data
    split_point = int(0.9*data.shape[0])

    train_x = data[0:split_point]
    train_y = targets[0:split_point]

    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    train_y = train_y.reshape(-1,1)

    #initialize model
    trans_model = TRANS_Model(train_x.shape[1:])

    #train model
    trans_model.train(train_x, train_y)

    #set up test data
    test_x = data[split_point:]
    test_y = targets[split_point:]

    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
    test_y = test_y.reshape(-1,1)

    #evaluate
    eval = trans_model.evaluate(test_x, test_y)
    avg = np.average(targets[split_point:])
    
    #print(eval)
    print("Mean Squared Error (bytes): ", eval[1])
    print("Mean Absolute Error (bytes): ", eval[0])
    print("Average Testing Chunk Size (bytes)", avg)

    #outs = trans_model.predict(test_x)
    #print("")
    #print(outs)

if __name__ == '__main__':
    train_and_evaluate()