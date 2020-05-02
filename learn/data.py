import pickle
import numpy as np
from random import shuffle, random
import pdb
import cv2
from partition import Partition
from utils_learn import *
from vae import *

class Data(object):
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.load_vae()
        self.load_actions()
        self.split_data()
        self.create_data()

    def load_vae(self):
        self.vae_path = '/home/yutongyan/PycharmProjects/rl-learn/vae.json'
        self.z_vector_size = 512
        self.vae = VAEController(z_size=self.z_vector_size)
        self.vae.load(self.vae_path)


    def load_actions(self):
        self.clip_to_actions = {}
        with open(self.args.actions_file) as f:
            for line in f.readlines():
                line = line.strip()
                parts = line.split()
                clip_id = parts[0]
                actions = list(map(eval, parts[1:]))
                self.clip_to_actions[clip_id] = actions

    def compute_nonzero_actions(self, clip_id, r, s):
        clip_id = clip_id.strip()
        r, s = min(r, s), max(r, s)
        actions = self.clip_to_actions[clip_id][r:s]
        n_nonzero = sum([1 if (a>=2 and a<=5) else 0 for a in actions])
        return n_nonzero

    def create_action_vector(self, clip_id, r, s):
        clip_id = clip_id.strip()
        r, s = min(r, s), max(r, s)
        actions = self.clip_to_actions[clip_id][r:s]
        action_vector = []
        for i in range(N_ACTIONS):
            action_vector.append(sum(map(lambda x:1. if x == i else 0., actions)))
        action_vector = np.array(action_vector)
        action_vector /= np.sum(action_vector)
        return action_vector

    def create_state_vector(self, folder_id, frames):
        state_vector_concat = np.ndarray(shape=(5, 512), dtype=np.float32)
        for i, frame in enumerate(frames):
            load_dir = '/home/yutongyan/Downloads/atari-lang/' +str(folder_id) +'/' + str(frame)+ '.png'
            img = cv2.imread(load_dir, 1)
            img = cv2.resize(img, (64, 64))
            img = img.astype(np.float).reshape((64, 64, 3))
            latent = self.vae.encode(img)
            state_vector_concat[i] = latent
        state_vector_concat=state_vector_concat.reshape(-1)
        return state_vector_concat.tolist()


    def load_data(self):
        self.data = pickle.load(open(self.args.data_file, 'rb'), encoding='bytes')
        self.match_data_to_frame = pickle.load(open('/home/yutongyan/PycharmProjects/rl-learn/learn/clip_id_frame_match.pkl', 'rb'), encoding='bytes')

    def split_data(self):
        self.train_pool = []
        self.valid_pool = []
        self.train_frame_ids = []
        self.valid_frame_ids = []

        partition = Partition()

        train_clips = []
        valid_clips = []

        train_corpus = []

        for clip in self.data:
            side = partition.clip_id_to_side(clip['clip_id'])
            if side == 'L':
                frames_list = self.match_data_to_frame[clip['clip_id']]
                for frame in frames_list:
                    self.valid_pool.append(clip)
                    valid_clips.append(clip['clip_id'])
                    frame = frame[:-4].split('/')[1]
                    frame_min = frame.split('-')[0]
                    frame_max = frame.split('-')[1]
                    indices = np.random.randint(frame_min, frame_max, size=5)
                    indices = np.sort(indices)
                    self.valid_frame_ids.append(indices)
            elif side == 'R' or side == 'C':
                frames_list = self.match_data_to_frame[clip['clip_id']]
                for frame in frames_list:
                    self.train_pool.append(clip)
                    train_clips.append(clip['clip_id'])
                    train_corpus.append(clip['sentence'])
                    frame = frame[:-4].split('/')[1]
                    frame_min = frame.split('-')[0]
                    frame_max = frame.split('-')[1]
                    indices = np.random.randint(frame_min, frame_max, size=5)
                    indices = np.sort(indices)
                    self.train_frame_ids.append(indices)


    def create_data(self):
        self.valid_prob = 0.2
        n_valid_data = int(self.args.n_data * self.valid_prob)
        n_train_data = self.args.n_data - n_valid_data

        self.action_list_train, self.lang_list_train, \
            self.labels_list_train, self.state_list_train = \
            self.create_data_split(self.train_pool, n_train_data, self.train_frame_ids)
        self.action_list_valid, self.lang_list_valid, \
            self.labels_list_valid, self.state_list_valid = \
            self.create_data_split(self.valid_pool, n_valid_data, self.valid_frame_ids)
        self.train_data = list(zip(self.action_list_train, self.lang_list_train, \
            self.labels_list_train, self.state_list_train))
        self.valid_data = list(zip(self.action_list_valid, self.lang_list_valid, \
            self.labels_list_valid, self.state_list_valid))

        pickle.dump(self.train_data,
                    open('/home/yutongyan/PycharmProjects/rl-learn/learn/train_data_'+self.args.lang_enc+'.pkl', "wb"))

        pickle.dump(self.valid_data,
                    open('/home/yutongyan/PycharmProjects/rl-learn/learn/valid_data_'+self.args.lang_enc+'.pkl', "wb"))
        print('Done loading data.')

        self.mean = np.mean(self.action_list_train, axis=-1)
        self.std = np.std(self.action_list_train, axis=-1)

    def get_data_pt_cond(self, data_pt):
        cond = None
        if self.args.lang_enc == 'onehot':
            cond = data_pt['onehot']
        elif self.args.lang_enc == 'glove':
            cond = data_pt['glove']
        elif self.args.lang_enc == 'infersent':
            cond = data_pt['infersent']
        else:
            raise NotImplementedError
        return cond

    def create_data_split(self, pool, n, frame_ids):
        action_list = []
        lang_list = []
        labels_list = []
        state_list = []
        for i in range(n):
            clip = np.random.choice(len(pool))
            clip_no = eval((pool[clip]['clip_id'].split('_')[-1])[:-4])
            r = np.random.choice(TRAJ_LEN)
            s = np.random.choice(TRAJ_LEN)
            r, s = min(r, s), max(r, s)
            if self.compute_nonzero_actions(pool[clip]['clip_id'], r, s) >= 5:
                data_pt_cur = pool[clip]
            else:
                continue

            while True:
                clip_alt = np.random.choice(len(pool))
                if data_pt_cur['clip_id'] != pool[clip_alt]['clip_id']:
                    break

            cond = self.get_data_pt_cond(pool[clip])

            action_vector = self.create_action_vector(pool[clip]['clip_id'], r, s)
            folder_id = pool[clip]['clip_id'].split('/')[0]
            state_vector = self.create_state_vector(folder_id, frame_ids[clip])

            action_list.append(action_vector)
            state_list.append(state_vector)
            lang_list.append(cond)
            labels_list.append(1)

            if np.random.random() < 0.5:
                cond_alt = self.get_data_pt_cond(pool[clip_alt])
                folder_id = pool[clip_alt]['clip_id'].split('/')[0]
                state_vector_alt = self.create_state_vector(folder_id, frame_ids[clip_alt])
                state_list.append(state_vector_alt)
                action_list.append(action_vector)
                lang_list.append(cond_alt)
                labels_list.append(0)
            else:
                action_vector_alt = np.random.random(N_ACTIONS)
                action_vector_alt /= np.sum(action_vector_alt)
                action_list.append(action_vector_alt)
                state_list.append(state_vector)
                lang_list.append(cond)
                labels_list.append(0)

        action_list = np.array(action_list)

        lang_list = np.array(lang_list)
        labels_list = np.array(labels_list)
        state_list = np.array(state_list)
        return action_list, lang_list, labels_list, state_list


