import os
import shutil
import argparse
import time

import random
import tensorflow as tf
import numpy as np

from multi_mnist import read_and_decode
from multi_mnist import read_test_data

from air.air_number_bbox_location import AIRModel
from air import evaluation_detection
from utils.checkpoints import build_logger, pile_image

random.seed(0)
np.random.seed(1234)
tf.set_random_seed(1235)

EPOCHS = 300
BATCH_SIZE = 64
CANVAS_SIZE = 50
MAX_STEPS = 6

# it is assumed that frequencies of more rare
# summaries in {NUM, VAR, IMG} are divisible
# by the frequencies of more frequent ones
LOG_EACH_ITERATION = 20
TESET_EACH_ITERATION = 200
IMG_SUMMARIES_EACH_ITERATIONS = 500

SAVE_PARAMS_EACH_ITERATIONS = 10000
NUM_IMAGES_TO_SAVE = 64

DEFAULT_READER_THREADS = 4
DEFAULT_RESULTS_FOLDER = "Not Valid"

# parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--results-folder", default=DEFAULT_RESULTS_FOLDER)
parser.add_argument("-k", "-key", "--key", default="")
parser.add_argument("-gpu", "--gpu", default="-1")
parser.add_argument(
    "-gl", "-gamma_location", "--gamma_location", type=float, default=0.0)
parser.add_argument("-gn", "-gamma_number", "--gamma_number", type=float, default=0.0)
parser.add_argument(
    "-gne", "-gamma_number_element", "--gamma_number_element", type=float, default=0.0)
parser.add_argument("-gm", "-gamma_margin", "--gamma_margin", type=float, default=0.0)
parser.add_argument("-gb", "-gamma_bbox", "--gamma_bbox", type=float, default=0.0)
parser.add_argument("-gs", "-gamma_size", "--gamma_size", type=float, default=0.0)
parser.add_argument(
    "-zt", "-z_pres_tempture", "--z_pres_tempture", type=float, default=0.1)
parser.add_argument("-o", "--overwrite-results", type=int, choices=[0, 1], default=0)
parser.add_argument("-t", "--reader-threads", type=int, default=DEFAULT_READER_THREADS)
parser.add_argument("-dn", "--dig_num", type=str, default="02")
parser.add_argument("-dl", "--dig_location", type=str, default="")
parser.add_argument("-ds", "--dig_surfix", type=str, default="")
args = parser.parse_args()

if int(args.gpu) >= 0 and int(args.gpu) <= 7:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

NUM_OF_DIGITS_TRAIN = [int(c) for c in args.dig_num]
if args.dig_location not in ["", "right_half"]:
    raise ValueError("not valid location of digit: " + args.dig_location)

NUM_OF_DIGITS_TEST = NUM_OF_DIGITS_TRAIN
name_of_common_train = args.dig_location + args.dig_surfix
for item in NUM_OF_DIGITS_TRAIN:
    name_of_common_train += str(item)
name_of_common_test = name_of_common_train
TRAIN_DATA_FILE = "./data/multi_mnist_data/common{}.tfrecords".format(
    name_of_common_train)
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
    filename_queue = tf.train.string_input_producer([TRAIN_DATA_FILE],
                                                    num_epochs=EPOCHS)
    train_data, train_targets = read_and_decode(filename_queue, BATCH_SIZE, CANVAS_SIZE,
                                                args.reader_threads)

    # # placeholders for feeding the same test dataset to test model
    test_data = tf.placeholder(tf.float32, shape=[None, CANVAS_SIZE**2])
    test_targets = tf.placeholder(tf.int32, shape=[None])

models = []
model_inputs = [[train_data, train_targets], [test_data, test_targets]]

# def constrains_x_y(x, y, gamma_location=args.gamma_location):
#     if args.dig_location == "right_half" and gamma_location >= 1e-8:
#         loss = tf.maximum(0.5 - x, 0) * gamma_location
#         return loss
#     else:
#         return 0.0

fix_scale = True
# if args.dig_surfix in ["BBOX", "sharedsize"]:
#     fix_scale = False
# else:
#     fix_scale = True

# creating two separate models - for training and testing - with
# identical configuration and sharing the same set of variables
for i in range(2):
    print("Creating {0} model...".format("training" if i == 0 else "testing"))
    models.append(
        AIRModel(
            model_inputs[i][0],
            model_inputs[i][1],
            max_steps=MAX_STEPS,
            max_digits=MAX_STEPS,
            rnn_units=256,
            canvas_size=CANVAS_SIZE,
            windows_size=28,
            vae_latent_dimensions=50,
            vae_recognition_units=(512, 256),
            vae_generative_units=(256, 512),
            fix_scale_distribution=fix_scale,
            vae_prior_mean=0.0,
            vae_prior_variance=1.0,
            vae_likelihood_std=0.0,
            scale_hidden_units=64,
            shift_hidden_units=64,
            z_pres_hidden_units=64,
            z_pres_prior_log_odds=-0.01,
            z_pres_temperature=args.z_pres_tempture,
            stopping_threshold=0.9,
            learning_rate=1e-4,
            gradient_clipping_norm=1.0,
            cnn=False,
            cnn_filters=8,
            num_summary_images=NUM_IMAGES_TO_SAVE,
            train=(i == 0),
            reuse=(i == 1),
            scope="air",
            constrains_x_y=None,
            constrains_num=NUM_OF_DIGITS_TRAIN,
            constrains_num_gamma=args.gamma_number,
            constrains_margin_gamma=args.gamma_margin,
            constrains_num_element_gamma=args.gamma_number_element,
            constrains_bbox_gamma=args.gamma_bbox,
            constrains_sharesize_gamma=args.gamma_size,
            fix_steps=NUM_OF_DIGITS_TRAIN[0] if len(NUM_OF_DIGITS_TRAIN) == 1 else None,
            annealing_schedules={
                # "constrains_bbox_gamma": {
                #     "init": 1e-10,
                #     "min": 0.0,
                #     "max": args.gamma_bbox,
                #     "factor": 10,
                #     "iters": 800,
                #     "staircase": False,
                #     "log": False,
                # },
                # "learning_rate": {
                #     "init": 1e-3, "min": 1e-4,
                #     "factor": 0.5, "iters": 10000,
                #     "staircase": False
                # }
            },
        ))

train_model, test_model = models[0], models[1]
sym_gen_samples = test_model.generated_samples

# start the training process
with tf.Session() as sess:
    coord = tf.train.Coordinator()

    log.info("Initializing variables...")
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    log.info("Starting queue runners...")
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # writer = tf.summary.FileWriter(SUMMARIES_FOLDER, sess.graph)
    saver = tf.train.Saver(max_to_keep=3)

    # diagnostic summaries are fetched from the test model
    # num_summaries = tf.summary.merge(test_model.num_summaries)
    # var_summaries = tf.summary.merge(test_model.var_summaries)
    # img_summaries = tf.summary.merge(test_model.img_summaries)

    # gradient summaries are fetched from the training model
    # grad_summaries = tf.summary.merge(train_model.grad_summaries)

    log.info("Reading test set...")
    # reading the test dataset, to be used with test model for
    # computing all summaries throughout the training process
    test_images, test_num_digits, _, test_positions, test_bboxs, _ = read_test_data(
        TEST_DATA_FILE, shift_zero_digits_images=True)

    returned_image = sess.run(train_data)
    returned_image = np.reshape(returned_image, [-1, 50, 50, 1])
    tmp_image = np.ones([returned_image.shape[0], 52, 52, 1])
    tmp_image[:, :50, :50, :] = returned_image
    pile_image(
        tmp_image,
        os.path.join(SUMMARIES_FOLDER, "visualize_data.png"),
    )

    log.info("Training...\n")

    try:
        # beginning with step = 0 to capture all summaries
        # and save the initial values of the model parameters
        # before the actual training process has started
        step = 0
        logged_results = [[] for _ in range(len(train_model.log_variables_name))]
        # print(logged_results)
        while True:

            returned_value = sess.run([train_model.training, train_model.global_step] +
                                      train_model.log_variables_tens)
            step = returned_value[1]
            for ind in range(2, len(returned_value)):
                logged_results[ind - 2].append(returned_value[ind])

            if step % LOG_EACH_ITERATION == 0:
                line = "step:{:6d}\t".format(step)
                for name, value in zip(train_model.log_variables_name, logged_results):
                    line += "{}:{:.4f}\t".format(name, np.mean(value))
                log.info(line)
                logged_results = [
                    [] for _ in range(len(train_model.log_variables_name))
                ]

            if step % TESET_EACH_ITERATION == 0:
                logged_results_test = sess.run(
                    test_model.log_variables_tens + [
                        test_model.rec_scales, test_model.rec_shifts,
                        test_model.rec_num_digits
                    ],
                    feed_dict={
                        test_data: test_images,
                        test_targets: test_num_digits
                    },
                )
                inf_scale = logged_results_test[-3]
                inf_shifts = logged_results_test[-2]
                inf_number = logged_results_test[-1]
                evaluation_results = evaluation_detection.evaluation(
                    test_positions, test_bboxs, inf_shifts, inf_scale, inf_number)
                mprecision, mrecall, mgtiou, mdectiou, global_iou = evaluation_results

                log.info(
                    "test:{:6d}\tprecision:{}\trecall:{}\tgtIoU:{:.4f}\tdetectionIoU:{:.4f}\tglobal_iou:{:.4f}"
                    .format(step, mprecision, mrecall, mgtiou, mdectiou, global_iou))

                logged_results_test = logged_results_test[:-3]
                line = "test:{:6d}\t".format(step)
                for name, value in zip(test_model.log_variables_name,
                                       logged_results_test):
                    line += "{}:{:.4f}\t".format(name, np.mean(value))
                log.info(line)

            if step % IMG_SUMMARIES_EACH_ITERATIONS == 0:
                returned_image = sess.run(
                    test_model.visualized_image,
                    feed_dict={
                        test_data: test_images,
                        test_targets: test_num_digits
                    },
                )
                pile_image(
                    returned_image,
                    os.path.join(SUMMARIES_FOLDER, "visualize_{}.png".format(step)),
                )

                returned_image, accuracy_all = sess.run(
                    [test_model.visualized_image_all, test_model.accuracy_instance],
                    feed_dict={
                        test_data: test_images,
                        test_targets: test_num_digits
                    },
                )
                wrong_image = returned_image[accuracy_all == 0]
                if wrong_image.shape[0] > 0:
                    pile_image(
                        wrong_image[:100],
                        os.path.join(SUMMARIES_FOLDER,
                                     "visualize_{}_wrong.png".format(step)),
                    )
                gen_sample, gen_bbox = sess.run(
                    [test_model.generated_samples, test_model.generated_samples_bbox])
                pile_image(
                    gen_sample,
                    os.path.join(SUMMARIES_FOLDER, "visualize_gen{}.png".format(step)),
                )
                pile_image(
                    gen_bbox,
                    os.path.join(SUMMARIES_FOLDER,
                                 "visualize_genbbox{}.png".format(step)),
                )

            # saving parameters with configured frequency
            if step % SAVE_PARAMS_EACH_ITERATIONS == 0:
                saver.save(sess, MODELS_FOLDER + "air-model", global_step=step)

    except tf.errors.OutOfRangeError:
        logged_results_test = sess.run(
            test_model.log_variables_tens +
            [test_model.rec_scales, test_model.rec_shifts, test_model.rec_num_digits],
            feed_dict={
                test_data: test_images,
                test_targets: test_num_digits
            },
        )
        inf_scale = logged_results_test[-3]
        inf_shifts = logged_results_test[-2]
        inf_number = logged_results_test[-1]
        evaluation_results = evaluation_detection.evaluation(
            test_positions, test_bboxs, inf_shifts, inf_scale, inf_number)
        mprecision, mrecall, mgtiou, mdectiou, globaliou = evaluation_results

        log.info(
            "test:{:6d}\tprecision:{}\trecall:{}\tgtIoU:{:.4f}\tdetectionIoU:{:.4f}\tglobal_iou:{:.4f}"
            .format(step, mprecision, mrecall, mgtiou, mdectiou, globaliou))

        logged_results_test = logged_results_test[:-3]
        line = "test:{:6d}\t".format(step)
        for name, value in zip(test_model.log_variables_name, logged_results_test):
            line += "{}:{:.4f}\t".format(name, np.mean(value))
        log.info(line)

        returned_image = sess.run(
            test_model.visualized_image,
            feed_dict={
                test_data: test_images,
                test_targets: test_num_digits
            },
        )
        pile_image(
            returned_image,
            os.path.join(SUMMARIES_FOLDER, "visualize_{}.png".format("final")),
        )

        returned_image, accuracy_all = sess.run(
            [test_model.visualized_image_all, test_model.accuracy_instance],
            feed_dict={
                test_data: test_images,
                test_targets: test_num_digits
            },
        )
        wrong_image = returned_image[accuracy_all == 0]
        if wrong_image.shape[0] > 0:
            pile_image(
                wrong_image[:100],
                os.path.join(SUMMARIES_FOLDER,
                             "visualize_{}_wrong.png".format("final")),
            )
        gen_sample, gen_bbox = sess.run(
            [test_model.generated_samples, test_model.generated_samples_bbox])
        pile_image(
            gen_sample,
            os.path.join(SUMMARIES_FOLDER, "visualize_gen{}.png".format(step)),
        )
        pile_image(
            gen_bbox,
            os.path.join(SUMMARIES_FOLDER, "visualize_genbbox{}.png".format(step)),
        )

        log.info("\ntraining has ended\n")

    finally:
        coord.request_stop()
        coord.join(threads)
