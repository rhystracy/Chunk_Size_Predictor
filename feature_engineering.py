import time
import numpy as np
import pandas as pd
import glob, os, subprocess, shlex, shutil

def get_csvs():
    return [file for file in glob.glob("./CSVs/*.csv")]

def get_data_one(file):
    data = list()
    targets = list()
    chunks = get_chunk_sizes(csv)
    for i in range(chunks.shape[0]-1):
        data.append(chunks[i])
        targets.append(chunks[i+1])
    
    data = np.asarray(data)
    targets = np.asarray(targets)
    return data, targets

def make_dataset():
    data = list()
    targets = list()
    csvs = get_csvs()
    for csv in csvs:
        chunks = get_chunk_sizes(csv)
        for i in range(chunks.shape[0]-1):
            data.append(chunks[i])
            targets.append(chunks[i+1])
    
    data = np.asarray(data)
    targets = np.asarray(targets)
    return data, targets

def get_chunk_sizes(file):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #df = pd.read_csv('data/example_pcap.csv')
    df = pd.read_csv(file)

    #dataset = pd.get_dummies(dataset)
    #dataset = dataset.replace(np.nan, 0)
    #dataset = dataset.to_numpy()

    df = df[df['Protocol'] == "QUIC"] #only need quic packets bc these are the only packets for the video

    df = df.reset_index(drop=True) #reset df indices after dropping rows

    client = df['Source'][0] #client sends first request to server

    df = df[df['Destination'] == client] #only look at packets incoming to the client

    df = df.reset_index(drop=True) #reset df indices after dropping rows

    #print(df)


    #go through and calculate chunk sizes

    chunk_size_data = []

    chunk_size = 0
    init = 1
    prev_start_time = 0
    for i in range(len(df)):
        if(df['Info'][i] != "Protected Payload (KP0)"):
            continue
        else:
            if(prev_start_time == 0):
                prev_start_time = df['Time'][i]
            if(df['Time'][i]>prev_start_time + 0.3): #more than 0.3s since last packet, means new chunk
                chunk_size_data.append(chunk_size)
                chunk_size = 0
                prev_start_time = df['Time'][i]
            chunk_size += int(df['Length'][i])
            prev_start_time = df['Time'][i]

    chunk_size_data=np.asarray(chunk_size_data)
    #print(chunk_size_data)
    #print(chunk_size_data.shape)

    return chunk_size_data

if __name__ == '__main__':
    data, targets = make_dataset()
    print(data.shape)
    print(targets.shape)