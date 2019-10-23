import os
import shutil
import argparse
import time

import tensorflow as tf
import numpy as np

from multi_mnist import read_and_decode
from multi_mnist import read_test_data

from air.vae_baseline import VAEModel
from utils.checkpoints import build_logger

EPOCHS = 300
BATCH_SIZE = 64
CANVAS_SIZE = 50
MAX_STEPS = 6

# it is assumed that frequencies of more rare
# summaries in {NUM, VAR, IMG} are divisible
# by the frequencies of more frequent ones
LOG_EACH_ITERATION = 20
NUM_SUMMARIES_EACH_ITERATIONS = 50
VAR_SUMMARIES_EACH_ITERATIONS = 250
IMG_SUMMARIES_EACH_ITERATIONS = 500

GRAD_SUMMARIES_EACH_ITERATIONS = 100
SAVE_PARAMS_EACH_ITERATIONS = 10000
NUM_IMAGES_TO_SAVE = 60

DEFAULT_READER_THREADS = 4
DEFAULT_RESULTS_FOLDER_FLAG = "Not Valid"

# parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--results-folder", default=DEFAULT_RESULTS_FOLDER_FLAG)
parser.add_argument("-k", "-key", "--key", default="")
parser.add_argument("-o", "--overwrite-results", type=int, choices=[0, 1], default=0)
parser.add_argument("-t", "--reader-threads", type=int, default=DEFAULT_READER_THREADS)
parser.add_argument("-dn", "--dig_num", type=str, default="02")
parser.add_argument("-dl", "--dig_location", type=str, default="")
args = parser.parse_args()

if args.dig_num == "02":
    NUM_OF_DIGITS_TRAIN = [0, 2]
elif args.dig_num == "13":
    NUM_OF_DIGITS_TRAIN = [1, 3]
else:
    raise ValueError("not valid number of digit: " + args.dig_num)

if args.dig_location not in ["", "right_half"]:
    raise ValueError("not valid location of digit: " + args.dig_location)

NUM_OF_DIGITS_TEST = NUM_OF_DIGITS_TRAIN
name_of_common_train = args.dig_location
for item in NUM_OF_DIGITS_TRAIN:
    name_of_common_train += str(item)
name_of_common_test = name_of_common_train
TRAIN_DATA_FILE = "./data/multi_mnist_data/common{}.tfrecords".format(
    name_of_common_train
)
TEST_DATA_FILE = "./data/multi_mnist_data/test{}.tfrecords".format(name_of_common_test)


if args.results_folder == "Not Valid":
    args.results_folder = "./results/{time}-({file})_(train.{train}_test.{test})".format(
        file=__file__,
        train=name_of_common_train.replace("_", "."),
        test=name_of_common_test.replace("_", "."),
        time=time.strftime("%Y-%m-%d-%H-%M"),
    )

# removing existing results folder (with content), if configured so
# otherwise, appending next available sequence # to the folder name
args.results_folder += "_({})".format(args.key)
if os.path.exists(args.results_folder):
    if args.overwrite_results:
        shutil.rmtree(args.results_folder, ignore_errors=True)
    else:
        folder, i = args.results_folder, 0
        args.results_folder = "{}_{}".format(folder, i)
        while os.path.exists(args.results_folder):
            i += 1
            args.results_folder = "{}_{}".format(folder, i)

MODELS_FOLDER = args.results_folder + "/models/"
SUMMARIES_FOLDER = args.results_folder + "/summary/"
SOURCE_FOLDER = args.results_folder + "/source/"

# creating result directories
os.makedirs(args.results_folder)
os.makedirs(MODELS_FOLDER)
os.makedirs(SUMMARIES_FOLDER)
os.makedirs(SOURCE_FOLDER)
log = build_logger(args.results_folder, args)

# creating a copy of the current version of *.py source files
for folder in ["./", "air/"]:
    destination = SOURCE_FOLDER
    if folder != "./":
        destination += folder
        os.makedirs(destination)
    for file in [f for f in os.listdir(folder) if f.endswith(".py")]:
        shutil.copy(folder + file, destination + file)

log.info("Creating input pipeline...")
with tf.variable_scope("pipeline"):
    # fetching a batch of numbers of digits and images from a queue
    filename_queue = tf.train.string_input_producer(
        [TRAIN_DATA_FILE], num_epochs=EPOCHS
    )
    train_data, train_targets = read_and_decode(
        filename_queue, BATCH_SIZE, CANVAS_SIZE, args.reader_threads
    )

    # # placeholders for feeding the same test dataset to test model
    test_data = tf.placeholder(tf.float32, shape=[None, CANVAS_SIZE ** 2])
    test_targets = tf.placeholder(tf.int32, shape=[None])

models = []
model_inputs = [[train_data, train_targets], [test_data, test_targets]]

# creating two separate models - for training and testing - with
# identical configuration and sharing the same set of variables
for i in range(2):
    print("Creating {0} model...".format("training" if i == 0 else "testing"))
    models.append(
        VAEModel(
            model_inputs[i][0],
            model_inputs[i][1],
            canvas_size=50,
            generation_unit=64,
            inference_unit=64,
            latent_unit=256,
            train=(i == 0),
            reuse=(i == 1),
            learning_rate=1e-3,
            gradient_clipping_norm=100.0,
            scope="vae_baseline",
        )
    )

train_model, test_model = models
sym_gen_samples = test_model.generated_samples

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# start the training process

with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()

    log.info("Initializing variables...")
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    log.info("Starting queue runners...")
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    writer = tf.summary.FileWriter(SUMMARIES_FOLDER, sess.graph)
    saver = tf.train.Saver(max_to_keep=3)

    # diagnostic summaries are fetched from the test model
    num_summaries = tf.summary.merge(test_model.num_summaries)
    # var_summaries = tf.summary.merge(test_model.var_summaries)
    img_summaries = tf.summary.merge(test_model.img_summaries)

    # gradient summaries are fetched from the training model
    # grad_summaries = tf.summary.merge(train_model.grad_summaries)
    grad_summaries = tf.no_op()

    log.info("Reading test set...")
    # reading the test dataset, to be used with test model for
    # computing all summaries throughout the training process
    test_images, test_num_digits, *_ = read_test_data(
        TEST_DATA_FILE, shift_zero_digits_images=True
    )

    log.info("Training...\n")

    try:
        # beginning with step = 0 to capture all summaries
        # and save the initial values of the model parameters
        # before the actual training process has started
        step = 0
        loss_list = []
        while True:
            summary_list = []
            if step % IMG_SUMMARIES_EACH_ITERATIONS == 0:
                summary_list.append(img_summaries)
            if step % NUM_SUMMARIES_EACH_ITERATIONS == 0:
                summary_list.append(num_summaries)
            if len(summary_list) > 0:
                summaries = sess.run(
                    summary_list,
                    feed_dict={test_data: test_images, test_targets: test_num_digits},
                )
                for item in summaries:
                    writer.add_summary(item, step)

            # saving parameters with configured frequency
            if step % SAVE_PARAMS_EACH_ITERATIONS == 0:
                saver.save(sess, MODELS_FOLDER + "air-model", global_step=step)

            # without gradient summaries
            _, train_loss, step = sess.run(
                [train_model.training, train_model.loss, train_model.global_step]
            )

            test_loss = sess.run(
                test_model.loss,
                feed_dict={test_data: test_images, test_targets: test_num_digits},
            )
            loss_list.append([train_loss, test_loss])

            if step % LOG_EACH_ITERATION == 0:

                l0, l1 = np.mean(loss_list[-LOG_EACH_ITERATION:], axis=0)
                log.info("iteration {}\ttrain loss {:.3f}".format(step, l0))
                log.info("iteration {}\ttest loss {:.3f}\t".format(step, l1))

    except tf.errors.OutOfRangeError:
        log.info("\ntraining has ended\n")

    finally:
        coord.request_stop()
        coord.join(threads)
