"""Load a trained network. Requires Python3. Put the files in the main method.

Also includes functionality to annotate the images, since I think that's needed
to show intuition in a paper.
"""
import os
import cv2
import sys
import time
import pickle
import functools
import tensorflow as tf
import numpy as np
from os.path import join
import baselines.common.tf_util as U
from baselines.imit.models import Actor

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)


class NetLoader:

    def __init__(self, net_file):
        """Trying to mirror the IMIT agent's loading method.

        When creating the actor, the TF is not actually created, for that we
        need placeholders and then to 'call' the actor.
        """
        self.actor = Actor(nb_actions=4, name='actor', network='cloth_cnn', use_keras=False)
        self.net_file = net_file

        # Exactly same as in the imit/imit_learner code, create actor network.
        self.observation_shape = (100, 100, 3)
        self.obs0 = tf.placeholder(tf.int32, shape=(None,)+self.observation_shape, name='obs0_imgs')
        self.obs0_f_imgs = tf.cast(self.obs0, tf.float32) / 255.0
        self.actor_tf = self.actor(self.obs0_f_imgs)

        # Handle miscellaneous TF stuff.
        self.sess = U.get_session()
        self.sess.run(tf.global_variables_initializer())
        _vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print('\nThe TF variables after init:\n')
        for vv in _vars:
            print('  {}'.format(vv))
        print('\nBaselines debugging:')
        U.display_var_info(_vars)
        print('\nNow let\'s call U.load_variables() ...')

        # Our imit code loads after initializing variables.
        U.load_variables(load_path=net_file, sess=self.sess)
        self.sess.graph.finalize()

    def forward_pass(self, img, reduce=True):
        """Run a forward pass. No other processing needed, I think.
        """
        feed = {self.obs0: [img]}
        result = self.sess.run(self.actor_tf, feed_dict=feed)
        if reduce:
            result = np.squeeze(result)
        return result

    def act_to_pixel(self, img, act, annotate=False, img_file=None):
        """Convert action to pixels, and optionally annotate to file.

        Returns the pick point (start) and target (ending) in pixels, but
        unfortunately w.r.t. the full image, and NOT the actual background
        plane, darn. But at least the forward pass itself seems to be working.
        """
        assert img.shape == self.observation_shape, img.shape
        assert img.shape[0] == img.shape[1], img.shape  # for now
        coord_min = 0
        coord_max = img.shape[0]

        # Convert from (-1,1) to the image pixels.  Note: this WILL sometimes
        # include points outside the range because we don't restrict that ---
        # but the agent should easily learn not to do that via IL or RL.
        pix_pick = (act[0] * 50 + 50,
                    act[1] * 50 + 50)
        pix_targ = ((act[0]+act[2]) * 50 + 50,
                    (act[1]+act[3]) * 50 + 50)

        # For image annotation we probably can just restrict to intervals. Also
        # convert to integers for drawing.
        pix_pick = ( int(max(min(pix_pick[0],coord_max),coord_min)),
                     int(max(min(pix_pick[1],coord_max),coord_min)) )
        pix_targ = ( int(max(min(pix_targ[0],coord_max),coord_min)),
                     int(max(min(pix_targ[1],coord_max),coord_min)) )

        if not annotate:
            return (pix_pick, pix_targ)

        # Now we annotate, save the image, and return pixels after all this.
        # AH ....we actually need this on the background plane. So it's harder
        # to interpret. Ack, we'll need a way to fit to that plane somehow ...
        # might be best to approximate it by hand?
        assert img_file is not None
        cv2.circle(img, center=pix_pick, radius=5, color=BLUE, thickness=1)
        cv2.circle(img, center=pix_targ, radius=3, color=RED, thickness=1)
        fname = img_file
        cv2.imwrite(filename=fname, img=img)

        return (pix_pick, pix_targ)


if __name__ == '__main__':
    # Adjust to load network, may be machine dependent, unfortunately.
    HEAD = '/home/seita/policies-cloth-sim/'
    POLICY = 'openai-2019-08-17-17-30-14-472186/checkpoints/00400'
    net_file = join(HEAD,POLICY)
    net_l = NetLoader(net_file)

    # Adjust to load images. I'm using ones from the demonstrator.
    IMG_HEAD = '/home/seita/gym-cloth/logs'
    image_files = sorted([join(IMG_HEAD,x) for x in os.listdir(IMG_HEAD) \
            if 'resized' in x and '.png' in x]
    )
    image_files = image_files[:20]

    # Where to save images (with annotated labels).
    save_img_path = join('data','tmp')
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path, exist_ok=True)

    # Iterate through images. We might as well save w/annotated images.
    for img_file in image_files:
        img = cv2.imread(img_file)
        assert img.shape == net_l.observation_shape, img.shape
        result = net_l.forward_pass(img)

        # Save the images, we may want to hav
        base = os.path.basename(os.path.normpath(img_file)).replace('resized_','')
        new_img_file = join(save_img_path, base)
        pixels = net_l.act_to_pixel(img=img,
                                    act=result,
                                    annotate=True,
                                    img_file=new_img_file)
        ps, pe = pixels
        print('{}    \t {} ---> {},     from {}'.format(result, ps, pe, img_file))
