import os
import tensorflow as tf
import numpy as np

from air.air_prior_correlated_reuse import AIRModel
# from air.air_model import AIRModel
from utils.checkpoints import plot_image

CANVAS_SIZE = 50
WINDOW_SIZE = 28

MODEL_FOLDER = "mnist_model_prior_trainright_half13_testright_half13-March-05-17-01reuse_Gnetwork"
MODEL_ROOT = os.path.join("./results/", MODEL_FOLDER)

test_data = tf.placeholder(tf.float32, shape=[None, CANVAS_SIZE**2])
test_targets = tf.placeholder(tf.int32, shape=[None])

print("Creating model...")
air_model = AIRModel(
    test_data,
    test_targets,
    max_steps=1,
    rnn_units=256,
    canvas_size=CANVAS_SIZE,
    windows_size=WINDOW_SIZE,
    vae_latent_dimensions=50,
    vae_recognition_units=(512, 256),
    vae_generative_units=(256, 512),
    vae_likelihood_std=0.3,
    scale_hidden_units=64,
    shift_hidden_units=64,
    z_pres_hidden_units=64,
    z_pres_temperature=1.0,
    stopping_threshold=0.99,
    cnn=False,
    train=False,
    reuse=False,
    scope="air",
    generation_batch_size=100,
)
sym_gen_samples = air_model.generated_samples

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    print("Restoring model...")
    ckpt_path = tf.train.latest_checkpoint(os.path.join(MODEL_ROOT, "models"))
    tf.train.Saver().restore(sess, ckpt_path)

    for i in range(4):
        gen_sample = sess.run(
            sym_gen_samples, feed_dict={air_model.max_steps_generation_placeholder: i})
        plot_image(
            np.reshape(gen_sample, [-1, 50, 50, 1]),
            os.path.join(MODEL_ROOT, 'gen_{}.png'.format(i)))
