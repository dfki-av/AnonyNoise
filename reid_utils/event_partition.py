# ------------------------------------------------------------------------------
# This script is based on the implementation described in:
# Person Re-Identification without Identification via Event Anonymization, by Shafiq Ahmad, Pietro Morerio, Alessio Del Bue, ICCV 2023
# 
# Original implementation: https://github.com/IIT-PAVIS/ReId_without_Id
# ------------------------------------------------------------------------------

import numpy as np
import pdb
import os
import time
import signal
import argparse


def load_events(event_dir):
    """Load Events"""
    event_path = event_dir + "/events.txt"
    infile = open(event_path, 'r')
    t = []
    x = []
    y = []
    p = []
    for line in infile:
        words = line.split()
        if len(words) == 2:
            print("first line")

        else:
            t.append(float(words[0]))
            x.append(int(words[1]))
            y.append(int(words[2]))
            p.append(int(words[3]))
    infile.close()
    return t, x, y, p

 
def id_cam(path):
    """extract ID and camera Number"""
    id_ = path.split("/")[-2]
    cam_ = path.split("/")[-1]
    cam = 'c' + cam_[4]
    
    return id_, cam, cam_


def constant_time(complete_data_path, integ_time, out_dir):

    ts, x, y, pol = load_events(complete_data_path)
    id_, cam, cam_ = id_cam(complete_data_path)
    
    save_dir = out_dir + f'/Event_ReId_v2/{id_}/{cam_}/'
    
    start_ts = ts[0]
    last_ts = ts[-1]
    #num_of_voxel = int((last_ts - start_ts) / integ_time)
            
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    voxel_count = 0
    """write first voxel, file name e.g. 001_c1_001.txt"""
    out_file = save_dir + id_ + "_" + cam + "_" + str("{:03d}".format(voxel_count+1)) + ".txt"
    event_chunk = open(out_file, "w")
    end_ts = ts[0] + integ_time  # integration time => 33.3ms

    for i in range(len(ts)):
        if ts[i] <= end_ts:
            event_chunk.write(str(ts[i]) + " " + str(x[i]) + " " + str(y[i]) + " " + str(pol[i]) + "\n")

        else:
            event_chunk.close()
            end_ts += integ_time
            voxel_count += 1

            """write 2nd, 3rd ..... voxel, file name e.g. 001_c1_002.txt"""
            out_file = save_dir + id_ + "_" + cam + "_" + str("{:03d}".format(voxel_count+1)) + ".txt"
            event_chunk = open(out_file, "w")
            
    event_chunk.close()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./Event_ReId_v2/', help='event stream directory')
    parser.add_argument('--out_dir', type=str, default='./output/', help='event stream directory')
    params = parser.parse_args()

    for i in range(33):
        i_d = "{0:03}".format(i+1)
        data_path = params.input_dir + str(i_d)
        sub_path = os.listdir(data_path)

        for j in range(4):
            complete_data_path = os.path.join(data_path, sub_path[j])
            constant_time(complete_data_path, 33.3, params.out_dir)

if __name__ == "__main__":
    main()