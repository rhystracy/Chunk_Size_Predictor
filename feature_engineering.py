import time
import numpy as np
import pandas as pd
import glob, os, subprocess, shlex, shutil

def get_csvs():
    return [file for file in glob.glob("./CSVs/*.csv")]

def get_csvs_duplicates():
    return [file for file in glob.glob("./CSVs_duplicates/*.csv")]

def get_data_one(csv, num_prev_chunks, num_prev_periods, period):
    data_chunks = list()
    targets_chunks = list()
    data_burstrate = list()
    targets_burstrate = list()
    chunks = get_chunk_sizes(csv, num_prev_chunks)
    try:
        chunks = get_chunk_sizes(csv, num_prev_chunks)
        for i in range(chunks.shape[0]-1):
            targets_chunks.append(chunks[i+1][num_prev_chunks-1])
            data_chunks.append(chunks[i])
    except pd.errors.ParserError:
        print("EOF error:", csv)
    except IndexError:
        print("Index Error:", csv, "doesn't have at least", num_prev_chunks+1, "chunks")
    
    try:
        burstrates = get_burstrates(csv, num_prev_periods, period)
        for i in range(burstrates.shape[0]-1):
            targets_burstrate.append(burstrates[i+1][num_prev_periods-1])
            data_burstrate.append(burstrates[i])
    except pd.errors.ParserError:
        print("EOF error:", csv)
    except IndexError:
        print("Index Error:", csv, "doesn't have at least", num_prev_periods+1, "periods")
    
    data_chunks = np.asarray(data_chunks)
    targets_chunks = np.asarray(targets_chunks)

    data_burstrate = np.asarray(data_burstrate)
    targets_burstrate = np.asarray(targets_burstrate)

    return data_chunks, targets_chunks, data_burstrate, targets_burstrate

def make_dataset(num_prev_chunks, num_prev_periods, period):
    data_chunks = list()
    targets_chunks = list()
    data_burstrate = list()
    targets_burstrate = list()
    csvs = get_csvs()
    num_csvs = len(csvs)
    a=0
    cur_percent = 0
    for csv in csvs:
        if(int(100*(a/num_csvs)) >= cur_percent):
            print(cur_percent, "percent done")
            cur_percent+=10
        a+=1

        '''try:
            chunks = get_chunk_sizes(csv, num_prev_chunks)
            for i in range(chunks.shape[0]-1):
                targets_chunks.append(chunks[i+1][num_prev_chunks-1])
                data_chunks.append(chunks[i])
        except pd.errors.ParserError:
            print("EOF error:", csv)
        except IndexError:
            print("Index Error:", csv, "doesn't have at least", num_prev_chunks+1, "chunks")
        except KeyError:
            print("Key Error:", csv, "no quic packets, can't get client ip")
        '''
        try:
            burstrates = get_burstrates(csv, num_prev_periods, period)
            for i in range(burstrates.shape[0]-1):
                targets_burstrate.append(burstrates[i+1][num_prev_periods-1])
                data_burstrate.append(burstrates[i])
        except pd.errors.ParserError:
            print("EOF error:", csv)
        except IndexError:
            print("Index Error:", csv, "doesn't have at least", num_prev_periods+1, "periods")
        except KeyError:
            print("Key Error:", csv, "no quic packets, can't get client ip")
    
    data_chunks = np.asarray(data_chunks)
    targets_chunks = np.asarray(targets_chunks)

    data_burstrate = np.asarray(data_burstrate)
    targets_burstrate = np.asarray(targets_burstrate)

    print("100 percent done")

    return data_chunks, targets_chunks, data_burstrate, targets_burstrate

def make_dataset_duplicates(num_prev_chunks, num_prev_periods, period):
    data_chunks = list()
    targets_chunks = list()
    data_burstrate = list()
    targets_burstrate = list()
    csvs = get_csvs_duplicates()
    num_csvs = len(csvs)
    a=0
    cur_percent = 0
    for csv in csvs:
        if(int(100*(a/num_csvs)) >= cur_percent):
            print(cur_percent, "percent done")
            cur_percent+=10
        a+=1

        '''try:
            chunks = get_chunk_sizes(csv, num_prev_chunks)
            for i in range(chunks.shape[0]-1):
                targets_chunks.append(chunks[i+1][num_prev_chunks-1])
                data_chunks.append(chunks[i])
        except pd.errors.ParserError:
            print("EOF error:", csv)
        except IndexError:
            print("Index Error:", csv, "doesn't have at least", num_prev_chunks+1, "chunks")
        except KeyError:
            print("Key Error:", csv, "no quic packets, can't get client ip")
        '''
        try:
            burstrates = get_burstrates(csv, num_prev_periods, period)
            for i in range(burstrates.shape[0]-1):
                targets_burstrate.append(burstrates[i+1][num_prev_periods-1])
                data_burstrate.append(burstrates[i])
        except pd.errors.ParserError:
            print("EOF error:", csv)
        except IndexError:
            print("Index Error:", csv, "doesn't have at least", num_prev_periods+1, "periods")
        except KeyError:
            print("Key Error:", csv, "no quic packets, can't get client ip")
    
    data_chunks = np.asarray(data_chunks)
    targets_chunks = np.asarray(targets_chunks)

    data_burstrate = np.asarray(data_burstrate)
    targets_burstrate = np.asarray(targets_burstrate)

    print("100 percent done")

    return data_chunks, targets_chunks, data_burstrate, targets_burstrate

def get_chunk_sizes(file, num_prev_chunks):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #df = pd.read_csv('data/example_pcap.csv')
    df = pd.read_csv(file)#, dtype={'frame.number':'int','frame.time':'object', 'frame.time_relative':'float',
                    #'frame.time_delta':'float', 'ip.src':'object', 'ip.dst':'object', 'ip.proto':'int', 'frame.len':'int',
                    #'frame.protocols':'object', 'data.len':'int', 'quic':'object'})

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
        if(len(prev_x) == num_prev_chunks):
            chunk_size_data.append(prev_x[:])
            prev_x.pop(0)
            prev_x.append(chunk_size_quic)
        else:
            prev_x.append(chunk_size_quic)
        if(len(prev_x) == num_prev_chunks):
            chunk_size_data.append(prev_x[:])
            prev_x.pop(0)
            prev_x.append(chunk_size_udp)
        else:
            prev_x.append(chunk_size_udp)
        if(len(prev_x) == num_prev_chunks):
            chunk_size_data.append(prev_x[:])
    else:
        if(len(prev_x) == num_prev_chunks):
            chunk_size_data.append(prev_x[:])
            prev_x.pop(0)
            prev_x.append(chunk_size_udp)
        else:
            prev_x.append(chunk_size_udp)
        if(len(prev_x) == num_prev_chunks):
            chunk_size_data.append(prev_x[:])
            prev_x.pop(0)
            prev_x.append(chunk_size_quic)
        else:
            prev_x.append(chunk_size_quic)
        if(len(prev_x) == num_prev_chunks):
            chunk_size_data.append(prev_x[:])

    chunk_size_data=np.asarray(chunk_size_data)

    #print(chunk_size_data)
    #print(chunk_size_data.shape)

    return chunk_size_data


def get_burstrates(file, num_prev_periods, period):
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


    burstrate_data = []
    prev_x = []
    num_time_periods = 1
    burstrate = 0
    num_zero_in_row = 0

    for i in range(len(df)):
        if(not("eth:ethertype:ip:udp:data" == df['frame.protocols'][i] or "eth:ethertype:ip:udp:quic" == df['frame.protocols'][i])):
            continue
        else:
            if(df['frame.time_relative'][i]>=num_time_periods*period): #more than 0.3s since last packet, means new chunk
                burstrate = burstrate/5 #get rate from total data in that 5s
                '''if(num_zero_in_row == 3):
                    break
                if(burstrate < 1000):
                    num_zero_in_row += 1
                else:
                    num_zero_in_row = 0'''
                if(len(prev_x) == num_prev_periods):
                    burstrate_data.append(prev_x[:])
                    prev_x.pop(0)
                    prev_x.append(burstrate)
                else:
                    prev_x.append(burstrate)
                burstrate = 0
                num_time_periods += 1
            else:
                burstrate += int(df['frame.len'][i])
                

    burstrate_data=np.asarray(burstrate_data)
    #print(burstrate_data)
    #print(burstrate_data.shape)
    
    return burstrate_data


if __name__ == '__main__':
    data_chunks, targets_chunks, data_burstrate, targets_burstrate = make_dataset(num_prev_chunks=5, num_prev_periods=5, period=5)
    print(data_chunks.shape)
    print(targets_chunks.shape)
    print(data_burstrate.shape)
    print(targets_burstrate.shape)