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
        # Daniel: we shouldn't be calling this if it's using our DDPG. Just in case.
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

    def __init__(self, limit, action_shape, observation_shape, dtype='float32'):
        """Daniel: careful about RAM usage. See:
        https://github.com/BerkeleyAutomation/baselines-fork/issues/9

        For this we can assume that in the replay buffer, the teacher samples
        come first, and are fixed ahead of time, so our 'starting' index for
        adding into the replay buffer should be offset by this quantity.
        """
        self.limit = limit
        self.observations0 = RingBuffer(limit, shape=observation_shape, dtype=dtype)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape, dtype=dtype)
        self.nb_teach = 0
        self.done_adding_teach = False

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        # TODO(Daniel): the -2 doesn't make sense, we don't need a proceeding
        # element because the next observation is in a separate ring buffer?? I
        # think it should be nb_entries, so we are in practice not sampling the
        # last two items in this replay buffer. I'm switching to -1, should do
        # 0 later if I'm confident we're not ignoring anything else ...
        batch_idxs = np.random.randint(self.nb_entries - 1, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        # Assume `x < self.nb_teach` (not equality!) is a teacher sample.
        flag_teacher = (batch_idxs < self.nb_teach).astype(np.float32)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
            'flag_teacher': array_min2d(flag_teacher),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, is_teacher=False,
               training=True):
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
        self.rewards.append(reward, is_teacher)
        self.observations1.append(obs1, is_teacher)
        self.terminals1.append(terminal1, is_teacher)

    def set_teacher_idx(self):
        """Call from DDPG+demos so we do not over-write teacher data.
        """
        self.done_adding_teach = True

    @property
    def nb_entries(self):
        return len(self.observations0)

    @property
    def nb_teach_entries(self):
        return self.nb_teach
