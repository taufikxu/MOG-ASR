import os
os.environ['PYTHONHASHSEED'] = str(0)
import shutil
import argparse
import time

import random
random.seed(0)
import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1235)

from multi_mnist import read_and_decode
from multi_mnist import read_test_data

from air.air_model import AIRModel
from air import evaluation_detection
from utils.checkpoints import build_logger, pile_image

EPOCHS = 300
BATCH_SIZE = 64
CANVAS_SIZE = 50
MAX_STEPS = 6

# it is assumed that frequencies of more rare
# summaries in {NUM, VAR, IMG} are divisible
# by the frequencies of more frequent ones
LOG_EACH_ITERATION = 20
TEST_EACH_ITERATION = 200
IMAGE_SAVE_ITERATION = 500

NUM_SUMMARIES_EACH_ITERATIONS = 50
VAR_SUMMARIES_EACH_ITERATIONS = 250
IMG_SUMMARIES_EACH_ITERATIONS = 500

# NUM_SUMMARIES_EACH_ITERATIONS = LOG_EACH_ITERATION
# VAR_SUMMARIES_EACH_ITERATIONS = LOG_EACH_ITERATION
# IMG_SUMMARIES_EACH_ITERATIONS = LOG_EACH_ITERATION

GRAD_SUMMARIES_EACH_ITERATIONS = 100
SAVE_PARAMS_EACH_ITERATIONS = 10000
NUM_IMAGES_TO_SAVE = 60

DEFAULT_READER_THREADS = 4
DEFAULT_RESULTS_FOLDER_FLAG = "Not Valid"

# parsing command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--results-folder", default=DEFAULT_RESULTS_FOLDER_FLAG)
parser.add_argument("-k", "-key", "--key", default="")
parser.add_argument("-gpu", "--gpu", default="-1")
parser.add_argument("-data", "--data", default="mnist")
parser.add_argument("-o", "--overwrite-results", type=int, choices=[0, 1], default=0)
parser.add_argument("-t", "--reader-threads", type=int, default=DEFAULT_READER_THREADS)
parser.add_argument("-dn", "--dig_num", type=str, default="02")
parser.add_argument("-dl", "--dig_location", type=str, default="")
parser.add_argument("-ds", "--dig_surfix", type=str, default="")
parser.add_argument("-ap", "--add_prior", type=str, default="")
args = parser.parse_args()

args.add_prior = args.add_prior.lower() in ["true", "t", "1"]

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

if args.data.lower() == "mnist":
    TRAIN_DATA_FILE = "./data/multi_mnist_data/common{}.tfrecords".format(
        name_of_common_train)
    TEST_DATA_FILE = "./data/multi_mnist_data/test{}.tfrecords".format(
        name_of_common_test)
    scale_prior_mean = -1.0
else:
    TRAIN_DATA_FILE = "./data/multi_dsprites/common{}.tfrecords".format(
        name_of_common_train)
    TEST_DATA_FILE = "./data/multi_dsprites/test{}.tfrecords".format(
        name_of_common_test)
    CANVAS_SIZE = 64
    scale_prior_mean = -1.0

if args.results_folder == "Not Valid":
    args.results_folder = "./results/{time}-({file}_{data})_(train.{train}_test.{test})".format(
        file=__file__,
        train=name_of_common_train.replace("_", "."),
        test=name_of_common_test.replace("_", "."),
        time=time.strftime("%Y-%m-%d-%H-%M"),
        data=args.data,
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
    # filename_queue_test = tf.train.string_input_producer([TEST_DATA_FILE])
    # test_data, test_targets = read_and_decode(
    #     filename_queue_test,
    #     BATCH_SIZE,
    #     CANVAS_SIZE,
    #     args.reader_threads,
    #     max_capacity=300)

models = []
model_inputs = [[train_data, train_targets], [test_data, test_targets]]

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
            scale_prior_mean=scale_prior_mean,
            scale_prior_variance=0.05,
            shift_prior_mean=0.0,
            shift_prior_variance=1.0,
            vae_prior_mean=0.0,
            vae_prior_variance=1.0,
            vae_likelihood_std=0.3,  # ours is 0.
            scale_hidden_units=64,
            shift_hidden_units=64,
            z_pres_hidden_units=64,
            z_pres_prior_log_odds=-0.01,
            z_pres_temperature=1.0,  # ours is 0.1
            stopping_threshold=0.99,  # ours is 0.9
            learning_rate=1e-4,
            gradient_clipping_norm=1.0,
            cnn=False,
            cnn_filters=8,
            num_summary_images=NUM_IMAGES_TO_SAVE,
            train=(i == 0),
            reuse=(i == 1),
            scope="air",
            annealing_schedules={
                "z_pres_prior_log_odds": {
                    "init": 10000.0,
                    "min": 0.000000001,
                    "factor": 0.1,
                    "iters": 3000,
                    "staircase": False,
                    "log": True,
                },
                # "learning_rate": {
                #     "init": 1e-3, "min": 1e-4,
                #     "factor": 0.5, "iters": 10000,
                #     "staircase": False
                # }
            },
            num_prior=NUM_OF_DIGITS_TRAIN if args.add_prior else None,
        ))

train_model, test_model = models[0], models[1]

# start the training process
with tf.Session() as sess:
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
    var_summaries = tf.summary.merge(test_model.var_summaries)
    img_summaries = tf.summary.merge(test_model.img_summaries)

    # gradient summaries are fetched from the training model
    grad_summaries = tf.summary.merge(train_model.grad_summaries)

    log.info("Reading test set...")
    # reading the test dataset, to be used with test model for
    # computing all summaries throughout the training process
    test_images, test_num_digits, _, test_positions, test_bboxs, _ = read_test_data(
        TEST_DATA_FILE, shift_zero_digits_images=True)

    log.info("Training...\n")
    min_loss, update_flag = 9999999., False
    best_elbo, best_accu, best_mse, best_iou = 0., 0., 0., 0.

    try:
        # beginning with step = 0 to capture all summaries
        # and save the initial values of the model parameters
        # before the actual training process has started
        step = 0
        loss_list, accu_list = [], []
        while True:
            # saving summaries with configured frequency
            if step % NUM_SUMMARIES_EACH_ITERATIONS == 0:
                if step % VAR_SUMMARIES_EACH_ITERATIONS == 0:
                    if step % IMG_SUMMARIES_EACH_ITERATIONS == 0:
                        num_sum, var_sum, img_sum = sess.run(
                            [num_summaries, var_summaries, img_summaries],
                            feed_dict={
                                test_data: test_images,
                                test_targets: test_num_digits,
                            },
                        )

                        writer.add_summary(img_sum, step)
                    else:
                        num_sum, var_sum = sess.run(
                            [num_summaries, var_summaries],
                            feed_dict={
                                test_data: test_images,
                                test_targets: test_num_digits,
                            },
                        )

                    writer.add_summary(var_sum, step)
                else:
                    num_sum = sess.run(
                        num_summaries,
                        feed_dict={
                            test_data: test_images,
                            test_targets: test_num_digits,
                        },
                    )

                writer.add_summary(num_sum, step)

            # saving parameters with configured frequency
            if step % SAVE_PARAMS_EACH_ITERATIONS == 0:
                saver.save(sess, MODELS_FOLDER + "air-model", global_step=step)

            # training step
            if step % GRAD_SUMMARIES_EACH_ITERATIONS == 0:
                # with gradient summaries
                _, train_loss, train_accuracy, train_mse, step, grad_sum = sess.run([
                    train_model.training,
                    train_model.loss,
                    train_model.accuracy,
                    train_model.mse_loss,
                    train_model.global_step,
                    grad_summaries,
                ])

                writer.add_summary(grad_sum, step)
            else:
                # without gradient summaries
                _, train_loss, train_accuracy, train_mse, step = sess.run([
                    train_model.training,
                    train_model.loss,
                    train_model.accuracy,
                    train_model.mse_loss,
                    train_model.global_step,
                ])

            loss_list.append([train_loss, train_accuracy, train_mse])

            if step % LOG_EACH_ITERATION == 0:

                l0, l1, l2 = np.mean(loss_list[-LOG_EACH_ITERATION:], axis=0)
                log.info(
                    "iteration {}\ttrain loss {:.3f}\ttrain accuracy {:.2f}, train mse {:.3f}"
                    .format(step, l0, l1, l2))
                if l0 < min_loss:
                    update_flag = True
                    min_loss = l0

            if step % TEST_EACH_ITERATION == 0:
                test_loss, test_accuracy, test_mse = sess.run(
                    [test_model.loss, test_model.accuracy, test_model.mse_loss],
                    feed_dict={
                        test_data: test_images,
                        test_targets: test_num_digits
                    })
                log.info(
                    "iteration {}\ttest loss {:.3f}\ttest accuracy {:.2f}, test mse {:.3f}"
                    .format(step, test_loss, test_accuracy, test_mse))

                logged_results_test = sess.run(
                    [
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
                    test_positions,
                    test_bboxs,
                    inf_shifts,
                    inf_scale,
                    inf_number,
                    csize=CANVAS_SIZE)
                mprecision, mrecall, mgtiou, mdectiou, globaliou = evaluation_results
                log.info(
                    "test:{:6d}\tprecision:{}\trecall:{}\tgtIoU:{:.4f}\tdetectionIoU:{:.4f}\tglobaliou:{:.4f}"
                    .format(step, mprecision, mrecall, mgtiou, mdectiou, globaliou))
                if update_flag is True:
                    update_flag = False
                    best_elbo = test_loss
                    best_accu = test_accuracy
                    best_mse = test_mse
                    best_iou = globaliou
                log.info("Current Best: elbo:{}\taccu:{}\tmse:{}\tiou:{}".format(
                    best_elbo, best_accu, best_mse, best_iou))

            if step % IMAGE_SAVE_ITERATION == 0:
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

                for i in NUM_OF_DIGITS_TRAIN:
                    gen_sample, gen_bbox = sess.run(
                        [
                            test_model.generated_samples,
                            test_model.generated_samples_bbox
                        ],
                        feed_dict={test_model.max_steps_generation_placeholder: i})
                    pile_image(
                        gen_sample,
                        os.path.join(SUMMARIES_FOLDER, "visualize_gen{}_{}.png".format(
                            step, i)))
                    pile_image(
                        gen_bbox,
                        os.path.join(SUMMARIES_FOLDER,
                                     "visualize_genbbox{}_{}.png".format(step, i)),
                    )

    except tf.errors.OutOfRangeError:
        test_loss, test_accuracy, test_mse = sess.run(
            [test_model.loss, test_model.accuracy, test_model.mse_loss],
            feed_dict={
                test_data: test_images,
                test_targets: test_num_digits
            })
        log.info(
            "iteration {}\ttest loss {:.3f}\ttest accuracy {:.2f}, test mse {:.3f}".
            format("final", test_loss, test_accuracy, test_mse))

        logged_results_test = sess.run(
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
            test_positions,
            test_bboxs,
            inf_shifts,
            inf_scale,
            inf_number,
            csize=CANVAS_SIZE)
        mprecision, mrecall, mgtiou, mdectiou, globaliou = evaluation_results
        log.info(
            "test:{}\tprecision:{}\trecall:{}\tgtIoU:{:.4f}\tdetectionIoU:{:.4f}\tglobaliou:{:.4f}"
            .format("final", mprecision, mrecall, mgtiou, mdectiou, globaliou))

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
        for i in NUM_OF_DIGITS_TRAIN:
            gen_sample, gen_bbox = sess.run(
                [test_model.generated_samples, test_model.generated_samples_bbox],
                feed_dict={test_model.max_steps_generation_placeholder: i})
            pile_image(
                gen_sample,
                os.path.join(SUMMARIES_FOLDER, "visualize_gen{}_{}.png".format(step,
                                                                               i)))
            pile_image(
                gen_bbox,
                os.path.join(SUMMARIES_FOLDER, "visualize_genbbox{}_{}.png".format(
                    step, i)),
            )
        log.info("Final Best: elbo:{}\taccu:{}\tmse:{}\tiou:{}".format(
            best_elbo, best_accu, best_mse, best_iou))
        log.info("\ntraining has ended\n")

    finally:
        coord.request_stop()
        coord.join(threads)

while (True):
    pass