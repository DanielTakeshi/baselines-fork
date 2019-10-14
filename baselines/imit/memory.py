"""Similar to DDPG except we only need obs and act, not the reward, etc.
"""
import numpy as np


class RingBuffer(object):

    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        if dtype == 'uint8':
            # Daniel: special case with our XP replay. Force memory allocation
            # right away by the += 0 op, to check that system has enough RAM.
            # Might not be good for speed so we'll have to time it.
            self.data = np.zeros((maxlen,) + shape, dtype=np.uint8)
            print("Allocating data of size {} ...".format(self.data.shape))
            self.data += 0
        else:
            self.data = np.zeros((maxlen,) + shape).astype(dtype)
        # Daniel: avoid over-writing teacher samples.
        self.teach_idx = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Daniel: we shouldn't be calling this if it's using our DDPG/IMIT.
        assert self.teach_idx == 0, \
            'Something went wrong, why are we calling this method?'
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        #return self.data[(self.start + idxs) % self.maxlen]
        # Daniel: seems like it's just fine to do this. It's the responsibility
        # of the caller to call a valid set of indices. And we do that with
        # randint in the memory class later. Here we avoid headaches with
        # `self.start` because I restrict it to be at least the teach_idx.
        return self.data[idxs]

    def append(self, v, is_teacher=False):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
            if is_teacher:
                self.teach_idx += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            #self.start = (self.start + 1) % self.maxlen
            self.start = max(self.teach_idx, (self.start + 1) % self.maxlen)
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):

    def __init__(self, limit, action_shape, observation_shape, dtype='float32',
                 do_valid=False):
        """Daniel: careful about RAM usage. See:
        https://github.com/BerkeleyAutomation/baselines-fork/issues/9

        For this we can assume that in the replay buffer, the teacher samples
        come first, and are fixed ahead of time, so our 'starting' index for
        adding into the replay buffer should be offset by this quantity.
        """
        self.limit = limit
        self.do_valid = do_valid
        if self.do_valid:
            self.valid_frac = 0.2
            self.nb_valid_items = 0  # will adjust later
        self.observations0 = RingBuffer(limit, shape=observation_shape, dtype=dtype)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.nb_teach = 0
        self.done_adding_teach = False

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        # TODO(Daniel): the -2 doesn't make sense, we don't need a proceeding
        # element because the next observation is in a separate ring buffer?? I
        # think it should be nb_entries, so we are in practice not sampling the
        # last two items in this replay buffer. I'm switching to -1, should do
        # 0 later if I'm confident we're not ignoring anything else ...
        if self.do_valid:
            # If we're doing validation, which should NOT normally be true,
            # ignore the first few items, which we assign to be in validation.
            batch_idxs = np.random.randint(self.nb_valid_items,
                                           self.nb_entries-1,
                                           size=batch_size)
        else:
            batch_idxs = np.random.randint(self.nb_entries-1, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)

        # Assume `x < self.nb_teach` (not equality!) is a teacher sample.
        flag_teacher = (batch_idxs < self.nb_teach).astype(np.float32)

        result = {
            'obs0': array_min2d(obs0_batch),
            'actions': array_min2d(action_batch),
            'flag_teacher': array_min2d(flag_teacher),
        }
        return result

    def append(self, obs0, action, is_teacher=False, training=True):
        """Keep separate copies of obs0, obs1. So it's not memory efficient.
        """
        if not training:
            return
        if is_teacher:
            assert not self.done_adding_teach, self.nb_teach
            assert self.nb_teach < self.limit, self.nb_teach
            self.nb_teach += 1
        self.observations0.append(obs0, is_teacher)
        self.actions.append(action, is_teacher)

    def set_teacher_idx(self):
        """Call from IMIT so we do not over-write teacher data.
        """
        self.done_adding_teach = True

    def set_valid_idx(self):
        """Set the validation index.
        """
        assert self.done_adding_teach
        self.nb_valid_items = int(self.valid_frac * self.nb_entries)

    @property
    def nb_entries(self):
        return len(self.observations0)

    @property
    def nb_teach_entries(self):
        return self.nb_teach

    @property
    def nb_valid(self):
        return self.nb_valid_items

    def get_valid_obs(self, s_idx, e_idx):
        """Get a validation minibatch with fixed starting and ending indices.
        """
        assert self.do_valid
        batch_idxs = np.arange(s_idx, e_idx)
        obs0_batch = self.observations0.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        result = {
            'obs0': array_min2d(obs0_batch),
            'actions': array_min2d(action_batch),
        }
        return result
