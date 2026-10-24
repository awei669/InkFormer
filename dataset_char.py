import os
import numpy as np
import torch
import torch.distributed
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
from torchvision import transforms
from PIL import Image

from utils import *



class Dataset_Char(Dataset):
    def __init__(self, 
                 args=None, 
                 ):
        '''
        self.hw will be format like:
        - user id (start from 0)
            - char id (GB2312 code)
                - handwritting char (np.array)
        ---
        self.std will be format like:
        - char id (GB2312 code)
            - standard char (np.array)
        ---
        Every char has been limited in [0,1]
        '''
        if args is None:
            self.format = 3
            self.n_sty = 8
        else:
            self.format = args.format
            self.n_sty = args.n_reference
        self.Nmax = 128

        self.hw = {}
        self.std = {}
        self.lookup_idx2ori = {}
        self.contained_chars = set()

    def load_hw(self, hw_path):
        '''load the handwritting characters from npy file.'''
        os.write(1, f'loading handwritting characters from {hw_path}...\n'.encode())
        data = np.load(hw_path, allow_pickle=True).item()

        # check format
        new_format = list((list(data.values())[0]).values())[0].shape[-1]
        assert self.format == new_format, 'the format of the handwritting characters is not consistent.'

        # update Nmax
        self.find_Nmax(data)

        key_last = list(self.hw.keys())[-1] if len(self.hw) > 0 else -1
        new_keys = list(range(key_last+1, key_last+1+len(data)))
        self.hw.update(dict(zip(new_keys, list(data.values()))))
        self.lookup_idx2ori.update(dict(zip(new_keys, list(data.keys()))))
    
    def switch_lookup(self, lookup_new): 
        '''switch the lookup table from old to new, at the same time switch the index of hw.

        - lookup_new: {index: ori_name}'''
        hw_new = {}
        lookup_new_inv = {v: k for k, v in lookup_new.items()}  # {ori_name: index}
        for k, v in self.hw.items():
            ori = self.lookup_idx2ori[k]
            idx_new = lookup_new_inv[ori]
            hw_new[idx_new] = v
        self.hw = hw_new
        self.lookup_idx2ori = lookup_new
        return 
    
    def hw_total(self):
        return np.sum([len(v) for v in self.hw.values()])
    
    def __len__(self):
        return len(self.hw)

    def load_std(self, std_path):
        '''load the standard characters from npy file.'''
        os.write(1, f'loading standard characters from {std_path}...\n'.encode())
        data = np.load(std_path, allow_pickle=True).item()

        # check format
        new_format = list(data.values())[0].shape[-1]
        assert self.format == new_format, 'the format of the standard characters is not consistent.'

        # update Nmax
        self.find_Nmax(data)

        self.std.update(data)
    
    def std_total(self):
        return len(self.std)

    def find_Nmax(self, data):
        for value in data.values():
            if isinstance(value, dict):
                self.find_Nmax(value)
            elif isinstance(value, np.ndarray):
                self.Nmax = max(self.Nmax, value.shape[0])
            else:
                raise ValueError('the data is not correctly formatted.')
        self.Nmax = 2 ** int(np.ceil(np.log2(self.Nmax)))   # make Nmax to be 2^n

    def update_contained_chars(self):
        for user in self.hw.values():
            self.contained_chars.update(user.keys())
        # check if std chars cover all the contained chars
        if len(self.contained_chars - set(self.std.keys())) > 0:
            raise ValueError('the standard characters do not cover all the handwritting characters.')   

    def __getitem__(self, uk_tuple):
        user, key = uk_tuple
        user_chars = self.hw[user]
        picked_gt_chars = user_chars[key]
        if key in self.std:
            picked_std_chars = self.std[key]
        else:
            picked_std_chars = np.zeros_like(picked_gt_chars)
        
        # pickup style reference
        picked_key = np.random.choice(list(user_chars.keys()), self.n_sty, replace=False)
        picked_sty_chars = [user_chars[k] for k in picked_key]
        if self.std:
            picked_sty_stds = [self.std[k] for k in picked_key]
        else:
            picked_sty_stds = [np.zeros_like(char) for char in picked_sty_chars]

        # corrdinate to velocity
        picked_gt_chars = x2v(picked_gt_chars)
        picked_std_chars = x2v(picked_std_chars)
        picked_sty_chars = [x2v(char) for char in picked_sty_chars]
        picked_sty_stds = [x2v(char) for char in picked_sty_stds]

        return (
            user, 
            key, 
            picked_gt_chars,
            picked_std_chars,
            picked_sty_chars, picked_sty_stds,
            )


class Train_Batch_Sampler(BatchSampler):
    def __init__(self, args, dset):
        self.batch_user = args.batch_user
        self.sample_per_user = args.sample_per_user

        self.user2key = {}
        for user in dset.hw.keys():
            self.user2key[user] = list(dset.hw[user].keys())
        self.users = list(dset.hw.keys())

        self.sampler = Sampler_everyone(self.user2key)  # BatchSampler要求的，其实没用到

    def __iter__(self):
        self.users = np.random.permutation(self.users)
        for i in range(0, len(self.users), self.batch_user):
            batch = []
            batch_users = self.users[i:i+self.batch_user]
            for user in batch_users:
                keys = np.random.choice(self.user2key[user], self.sample_per_user, replace=False)
                batch.extend([(user, key) for key in keys])
            yield batch

    def __len__(self):
        return int((len(self.users)+self.batch_user-1) / self.batch_user)


class collate_fn:
    def __init__(self, Nmax, format=3):
        self.Nmax = Nmax
        self.format = format

    def padding(self, chars):
        '''padding the characters to the Nmax length using:
        - format 3: (0,0,-1)
        - format 5: (0,0,0,0,1)
        '''
        mask = np.zeros((len(chars), self.Nmax))
        for i in range(len(chars)):
            char = chars[i]
            if len(char) < self.Nmax:
                pad = np.zeros((self.Nmax-len(char), self.format))
                if self.format == 3:
                    pad[:,-1] = -1
                    chars[i] = np.concatenate([char, pad], axis=0)
                elif self.format == 5:
                    pad[:,-1] = 1
                    chars[i] = np.concatenate([char, pad], axis=0)
            else: pass
            mask[i, :len(char)] = 1
            assert len(chars[i]) == self.Nmax
        return np.stack(chars, axis=0), mask

    def resampling(self, chars):
        '''resample the characters to the Nmax length using linear interpolation. '''
        resampled = []
        if self.format == 3:
            for char in chars:
                t = np.linspace(0, 1, len(char))
                t_new = np.linspace(0, 1, self.Nmax)
                x = np.interp(t_new, t, char[:,0])
                y = np.interp(t_new, t, char[:,1])
                # 对于ps，我们要保证pen-up前的点是valid，但pen-up至下一个pen-down的点是invalid。通过计算Δps就可以实现
                ps = np.interp(t_new, t, char[:,2])
                delta_ps = np.diff(ps) ; delta_ps = np.concatenate([[0], delta_ps], axis=0)
                ps = np.where(delta_ps>0, -1, 1)  # Δps>0, pen-up; Δps<=0, pen-down
                resampled.append(np.stack([x,y,ps], axis=-1))
            return np.stack(resampled, axis=0), np.ones((len(resampled), self.Nmax))
        else:
            # TODO: implement the resampling for format=5
            raise NotImplementedError

    def length_regular(self, chars):
        return self.padding(chars)
        # return self.resampling(chars)

    def batch_Nmax(self, chars_lists):
        '''calculate the Nmax of the batch.'''
        Nmax = []
        for chars in chars_lists:
            Nmax.append(max([char.shape[0] for char in chars]))
        Nmax = max(Nmax)
        Nmax = 8 * np.ceil(Nmax / 8)        # makes length efficient for 3-times down-sampling
        return int(Nmax)
        
    def __call__(self, batch):
        '''collate function for DataLoader.'''
        writer_id, keys, \
        hw_chars, std_chars, \
        sty_refs, sty_stds, \
                            = zip(*batch)
        writer_id = np.stack(writer_id)
        writer_id = torch.from_numpy(writer_id).long()
        keys = np.stack(keys, dtype=object)
        hw_chars = list(hw_chars)
        std_chars = list(std_chars)
        sty_refs = [char for chars in sty_refs for char in chars]
        sty_stds = [char for chars in sty_stds for char in chars]
        
        self.Nmax = self.batch_Nmax([hw_chars, std_chars, sty_refs, sty_stds])
        
        hw_chars, hw_mask = self.length_regular(hw_chars)
        hw_chars = torch.from_numpy(hw_chars).float() 
        hw_mask = torch.from_numpy(hw_mask).float()
        std_chars, std_mask = self.length_regular(std_chars)
        std_chars = torch.from_numpy(std_chars).float() 
        std_mask = torch.from_numpy(std_mask).float()

        sty_refs, sty_mask = self.length_regular(sty_refs)
        sty_refs = torch.from_numpy(sty_refs).float()
        sty_mask = torch.from_numpy(sty_mask).float()
        sty_stds, sty_stds_mask = self.length_regular(sty_stds)
        sty_stds = torch.from_numpy(sty_stds).float()
        sty_stds_mask = torch.from_numpy(sty_stds_mask).float()

        B = len(writer_id) ; n = len(sty_refs) // B
        sty_refs = sty_refs.reshape(B, n, *sty_refs.shape[1:])
        sty_mask = sty_mask.reshape(B, n, *sty_mask.shape[1:])
        sty_stds = sty_stds.reshape(B, n, *sty_stds.shape[1:])
        sty_stds_mask = sty_stds_mask.reshape(B, n, *sty_stds_mask.shape[1:])

        batched_data = {
            'writer_id': writer_id,
            'keys': keys,
            'hw_chars': hw_chars, 'hw_mask': hw_mask,
            # 'hw_imgs': hw_imgs,
            'std_chars': std_chars, 'std_mask': std_mask,
            # 'std_imgs': std_imgs,
            'sty_refs': sty_refs, 'sty_mask': sty_mask,
            'sty_stds': sty_stds, 'sty_stds_mask': sty_stds_mask,
        }
        return batched_data



##################################################
'''Dataset, sampler, collate_fn for inferencing'''

'''Using training dataset'''

class Test_Batch_Sampler(BatchSampler):
    def __init__(self, dset:Dataset_Char, batch_size:int=100):
        self.batch_size = batch_size

        self.user2key = {}
        for user in dset.hw.keys():
            self.user2key[user] = list(dset.hw[user].keys())

        self.sampler = Sampler_everyone(self.user2key)  # BatchSampler要求的，其实没用到

    def __iter__(self):
        ### chars of one user in one batch
        for u in self.user2key.keys():
            keys = self.user2key[u]
            for i in range(0, len(keys), self.batch_size):
                keys_b = keys[i:i+self.batch_size]
                batch = list(zip([u]*len(keys_b), keys_b))
                yield batch

    def __len__(self):
        n_batch = 0
        for u in self.user2key.keys():
            keys = self.user2key[u]
            n_batch += int(np.ceil(len(keys) / self.batch_size))
        return n_batch


class Sampler_everyone(Sampler):
    def __init__(self, user_keys_dict):
        self.user_keys_dict = user_keys_dict

    def __iter__(self):
        for u in self.user_keys_dict.keys():
            keys = self.user_keys_dict[u]
            for k in keys:
                yield (u, k)
    
    def __len__(self):
        return sum([len(keys) for keys in self.user_keys_dict.values()])


'''Using training collate_fn'''