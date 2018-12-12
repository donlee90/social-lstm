import torch

import torch.utils.data as data
import numpy as np
import pandas as pd


class TrajectoryDataset(data.Dataset):
    def __init__(self, data_files, max_length=20, min_length=10, to_tensor=True):

        # Max and min length of consecutive sequence
        self.max_length = max_length
        self.min_length = min_length

        # If true return torch.Tensor, or otherwise return np.array
        self.to_tensor = to_tensor

        # List containing raw data for each scene
        self.scenes = []

        # List of dictionary containing:
        #   "scene_id": (int)
        #   "start": (int)
        #   "interval": (int)
        self.frame_seqs = []

        # Each data file contains raw data from a single scene
        for scene_id, data_file in enumerate(data_files):
            scene = _load_data(data_file)
            interval = _get_frame_interval(scene)
            frame_ids = _get_frame_ids(scene)

            # For each frame in a scene,
            # check if there are at least min_length frames
            # starting from the current frame.
            for frame in frame_ids:
                start = frame
                end = start + (max_length * interval)

                # get data sequence between the "start" and "end" frames (inclusive)
                data_seq = _get_data_sequence_between(scene, start, end)

                # add the info about frame sequence to frame_seqs
                if len(_get_frame_ids(data_seq)) >= min_length:
                    self.frame_seqs.append({
                            "scene_id": scene_id,
                            "start": start,
                            "interval": interval
                        })

            self.scenes.append(scene)

    def __getitem__(self, index):
        """
        N : number of agents
        L : max sequence length
        D : dimension of spatial trajectory coordinate
        Return:
            source - tensor of shape (N, L, D)
            target - tensor of shape (N, L, D)
            mask - tensor of shape (N, L)
        """
        # Get frame sequence and scene
        frame_seq = self.frame_seqs[index]
        scene = self.scenes[frame_seq['scene_id']]
        interval = frame_seq['interval']

        start = frame_seq['start'] 
        end = start + ((self.max_length - 1) * interval)

        # Get source & target sequence data from the scene
        source_seq = _get_data_sequence_between(scene, start, end)
        target_seq = _get_data_sequence_between(scene, (start+interval), (end+interval))

        # Get union of agent_ids in source & target sequences
        source_agents = set(source_seq.agent_id.unique())
        target_agents = set(target_seq.agent_id.unique())
        agents = sorted(list(source_agents | target_agents))

        # Convert source & target sequence to arrays
        source, source_mask = _to_array(source_seq, agents, self.max_length,
                                    start, interval)
        target, target_mask = _to_array(target_seq, agents, self.max_length,
                                    (start+interval), interval)

        if (self.to_tensor):
            source = torch.Tensor(source)
            target = torch.Tensor(target)
            source_mask = torch.Tensor(source_mask)
            target_mask = torch.Tensor(target_mask)

        return source, target, source_mask, target_mask

    def __len__(self):
        return len(self.frame_seqs)


# Helper functions
def _load_data(data_file):
    """ Read data from a file and returns a pd.DataFrame """
    df = pd.read_csv(data_file, sep=" ", header=None)
    df.columns = ["frame_id", "agent_id", "x", "y"]
    return df

def _get_data_sequence_between(df, start, end):
    """ Returns all data with frame_id between start and end (inclusive) """
    return df.loc[df.frame_id.between(start, end)]

def _get_frame_interval(df):
    """ Calculate frame interval of the DataFrame df.
        Assumes that the first two frame_ids are consecutive
    """
    return df.frame_id[1] - df.frame_id[0]

def _get_frame_ids(df):
    """ Returns unique frame_ids in the DataFrame df """
    return df.frame_id.unique()

def _to_array(df, agents, max_length, start, interval, dim=2):
    """ Convert input DataFrame df to 3-dimensional Numpy array"""
    num_agents = len(agents)

    # First create an array of shape (N, L, D) filled with inf.
    # We will replace infs with 0 after we compute mask
    array = np.full((num_agents, max_length, dim), np.inf)

    # Compute indexs to fill out
    agent_idxs = (df.agent_id.apply(agents.index)).astype(int)
    frame_idxs = ((df.frame_id - start) / interval).astype(int)
    coords = df[['x', 'y']]
    assert len(coords) == len(frame_idxs) and len(coords) == len(agent_idxs)

    # Fill out arrays with coordinates
    array[agent_idxs, frame_idxs] = coords

    # Compute mask where mask[i, j] == 0 for array[i, j] == inf
    mask = (array != np.inf).astype(int)

    # Finally replace inf with 0
    array[array == np.inf] = 0

    return array, mask

