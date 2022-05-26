import time
import numpy as np
import pandas as pd
import glob, os, subprocess, shlex, shutil

def get_csvs():
    return [file for file in glob.glob("./CSVs/*.csv")]

def get_data_one(file, num_prev_chunks):
    data = list()
    targets = list()
    chunks = get_chunk_sizes(csv, num_prev_chunks)
    for i in range(chunks.shape[0]-1):
        data.append(chunks[i])
        targets.append(chunks[i+1])
    
    data = np.asarray(data)
    targets = np.asarray(targets)
    return data, targets

def make_dataset(num_prev_chunks):
    data = list()
    targets = list()
    csvs = get_csvs()
    for csv in csvs:
        chunks = get_chunk_sizes(csv, num_prev_chunks)
        for i in range(chunks.shape[0]-1):
            data.append(chunks[i])
            targets.append(chunks[i+1][num_prev_chunks-1])
    
    data = np.asarray(data)
    targets = np.asarray(targets)
    return data, targets

def get_chunk_sizes(file, num_prev_chunks):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #df = pd.read_csv('data/example_pcap.csv')
    df = pd.read_csv(file)

    #dataset = pd.get_dummies(dataset)
    #dataset = dataset.replace(np.nan, 0)
    #dataset = dataset.to_numpy()

    df_quic = df[df["quic"] == "quic"] #only need quic packets bc these are the only packets for the video

    df_quic = df_quic.reset_index(drop=True) #reset df indices after dropping rows

    client = df_quic['ip.src'][0] #client sends first request to server

    df = df[df['ip.dst'] == client] #only look at packets incoming to the client

    df = df.reset_index(drop=True) #reset df indices after dropping rows


    #go through and calculate chunk sizes

    chunk_size_data = []
    prev_x = []

    last = 0 #0=udp last, 1=quic last
    chunk_size_udp = 0
    chunk_size_quic = 0
    prev_start_time_udp = 0
    prev_start_time_quic = 0
    for i in range(len(df)):
        if(not("eth:ethertype:ip:udp:data" == df['frame.protocols'][i] or "eth:ethertype:ip:udp:quic" == df['frame.protocols'][i])):
            continue
        else:
            if(prev_start_time_udp == 0 and "eth:ethertype:ip:udp:data" == df['frame.protocols'][i]):
                prev_start_time_udp = df['frame.time_relative'][i]
            if(prev_start_time_quic == 0 and "eth:ethertype:ip:udp:quic" == df['frame.protocols'][i]):
                prev_start_time_quic = df['frame.time_relative'][i]
            if("eth:ethertype:ip:udp:data" == df['frame.protocols'][i] and df['frame.time_relative'][i]>prev_start_time_udp + 0.1): #more than 0.1s since last packet, means new chunk
                if(len(prev_x) == num_prev_chunks):
                    chunk_size_data.append(prev_x[:])
                    prev_x.pop(0)
                    prev_x.append(chunk_size_udp)
                else:
                    prev_x.append(chunk_size_udp)
                chunk_size_udp = 0
                prev_start_time_udp = df['frame.time_relative'][i]
                last = 0
            if("eth:ethertype:ip:udp:quic" == df['frame.protocols'][i] and df['frame.time_relative'][i]>prev_start_time_quic + 0.1): #more than 0.1s since last packet, means new chunk
                if(len(prev_x) == num_prev_chunks):
                    chunk_size_data.append(prev_x[:])
                    prev_x.pop(0)
                    prev_x.append(chunk_size_quic)
                else:
                    prev_x.append(chunk_size_quic)
                chunk_size_quic = 0
                prev_start_time_quic = df['frame.time_relative'][i]
                last = 1
            if("eth:ethertype:ip:udp:data" == df['frame.protocols'][i]):
                chunk_size_udp += int(df['frame.len'][i])
                prev_start_time_udp = df['frame.time_relative'][i]
            if("eth:ethertype:ip:udp:quic" == df['frame.protocols'][i]):
                chunk_size_quic += int(df['frame.len'][i])
                prev_start_time_quic = df['frame.time_relative'][i]

    if(last == 0):
        if(chunk_size_quic>0):
            chunk_size_data.append(prev_x[:])
            prev_x.pop(0)
            prev_x.append(chunk_size_quic)
        if(chunk_size_udp>0):
            chunk_size_data.append(prev_x[:])
            prev_x.pop(0)
            prev_x.append(chunk_size_udp)
        chunk_size_data.append(prev_x[:])
    else:
        if(chunk_size_udp>0):
            chunk_size_data.append(prev_x[:])
            prev_x.pop(0)
            prev_x.append(chunk_size_udp)
        if(chunk_size_quic>0):
            chunk_size_data.append(prev_x[:])
            prev_x.pop(0)
            prev_x.append(chunk_size_quic)
        chunk_size_data.append(prev_x)

    chunk_size_data=np.asarray(chunk_size_data)

    #print(chunk_size_data)
    #print(chunk_size_data.shape)

    return chunk_size_data

if __name__ == '__main__':
    data, targets = make_dataset(5)
    print(data.shape)
    print(targets.shape)