import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from NGSIM_env.data.ngsim import *
import pickle


def trajectory_process(trajectory, decision_time):
    trajectory.columns = ['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']
    trajectory['v_x'] = trajectory['x'].diff() / 0.1
    trajectory['v_y'] = trajectory['y'].diff() / 0.1
    trajectory['a_x'] = trajectory['v_x'].diff() / 0.1
    trajectory['a_y'] = trajectory['v_y'].diff() / 0.1

    trajectory = trajectory[trajectory['frame'] >= decision_time - 20]

    return trajectory


def build_trajecotry():
    with open('data/sample_data.pkl', "rb") as input_file:
        data = pickle.load(input_file)

    record_trajectory = {'lcv': [], 'fv': [], 'nlv': [], 'olv': []}
    
    for pair in data:
        end_frame = pair['end_frame'].values[0]
        decision_frame = pair['decision_frame'].values[0]
        lcv = pair[['frame', 'id_x', 'y_x', 'x_x', 'width_x', 'height_x', 'laneId_x']]
        fv = pair[['frame', 'id_y', 'y_y', 'x_y', 'width_y', 'height_y', 'laneId_y']]
        nlv = pair[['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']]
        olv = pair[['frame', 'id_z', 'y_z', 'x_z', 'width_z', 'height_z', 'laneId_z']]

        lcv = trajectory_process(lcv, decision_frame)
        fv = trajectory_process(fv, decision_frame)
        nlv = trajectory_process(nlv, decision_frame)
        olv = trajectory_process(olv, decision_frame)

        if lcv.isnull().values.any() or fv.isnull().values.any() or nlv.isnull().values.any() or olv.isnull().values.any():
            continue

        record_trajectory['lcv'].append(lcv)
        record_trajectory['fv'].append(fv)
        record_trajectory['nlv'].append(nlv)
        record_trajectory['olv'].append(olv)

    return record_trajectory
