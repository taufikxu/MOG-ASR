import tensorflow as tf
from tensorflow import layers


class VAEModel:

    def __init__(self,
                 input_images,
                 target_num_digits,
                 canvas_size=50,
                 generation_unit=64,
                 inference_unit=64,
                 latent_unit=256,
                 train=False,
                 reuse=False,
                 learning_rate=1e-3,
                 gradient_clipping_norm=100.0,
                 generation_batch_size=100,
                 scope="vae_baseline",
                 num_summary_images=60):

        self.input_images = input_images
        self.input_as_image = tf.reshape(input_images,
                                         [-1, canvas_size, canvas_size, 1])
        self.target_num_digits = target_num_digits
        self.batch_size = tf.shape(input_images)[0]

        self.canvas_size = canvas_size

        self.inference_unit = inference_unit
        self.generation_unit = generation_unit
        self.latent_unit = latent_unit

        self.learning_rate = learning_rate
        self.gradient_clipping_norm = gradient_clipping_norm
        self.train = train

        self.num_summaries = []
        self.img_summaries = []
        self.num_summary_images = num_summary_images

        with tf.variable_scope(scope, reuse=reuse):
            self.global_step = tf.get_variable(
                name="global_step",
                shape=[],
                dtype=tf.int32,
                initializer=tf.constant_initializer(0),
                trainable=False,
            )

            self.rec_num_digits = None
            self.rec_scales = None
            self.rec_shifts = None
            self.reconstruction = None
            self.loss = None
            self.accuracy = None
            self.training = None

            self._create_model()

        with tf.variable_scope(scope, reuse=True):
            self.generation_batch_size = generation_batch_size
            self.generated_samples = self._create_generation()

    @staticmethod
    def _create_annealed_tensor(param, schedule, global_step, eps=10e-10):
        value = tf.train.exponential_decay(
            learning_rate=schedule["init"],
            global_step=global_step,
            decay_steps=schedule["iters"],
            decay_rate=schedule["factor"],
            staircase=False if "staircase" not in schedule else schedule["staircase"],
            name=param,
        )

        if "min" in schedule:
            value = tf.maximum(value, schedule["min"], name=param + "_max")

        if "max" in schedule:
            value = tf.minimum(value, schedule["max"], name=param + "_min")

        if "log" in schedule and schedule["log"]:
            value = tf.log(value + eps, name=param + "_log")

        return value

    @staticmethod
    def _sample_from_mvn(mean, diag_variance, shape=None):
        # sampling from the multivariate normal
        # with given mean and diagonal covaraince
        tmp_shape = shape if shape is not None else tf.shape(mean)
        standard_normal = tf.random_normal(tmp_shape)
        return mean + standard_normal * tf.sqrt(diag_variance)

    def _inference_network(self, x, reuse=None):
        with tf.variable_scope("inference", reuse=reuse):
            feature = tf.reshape(x, [-1, self.canvas_size, self.canvas_size, 1])
            feature = layers.conv2d(feature, self.inference_unit, 3, padding="valid")
            feature = tf.nn.relu(feature)  # 48 * 48

            feature = layers.conv2d(feature, self.inference_unit, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.max_pooling2d(feature, 2, 2)  # 24 * 24

            feature = layers.conv2d(feature, self.inference_unit * 2, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.conv2d(feature, self.inference_unit * 2, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.max_pooling2d(feature, 2, 2)  # 12 * 12

            feature = layers.conv2d(feature, self.inference_unit * 4, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.conv2d(feature, self.inference_unit * 4, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.max_pooling2d(feature, 2, 2)  # 6 * 6

            feature = layers.conv2d(feature, self.inference_unit * 8, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.conv2d(feature, self.inference_unit * 8, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.average_pooling2d(feature, 6, 6)

            feature = layers.flatten(feature)
            mean = layers.dense(feature, self.latent_unit)
            log_var = layers.dense(feature, self.latent_unit)
            return mean, log_var

    def _generation_network(self, z, return_logits=True, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            feature = layers.dense(z, 6 * 6 * self.generation_unit * 8)
            feature = tf.nn.relu(feature)
            feature = tf.reshape(feature, [-1, 6, 6, self.generation_unit * 8])

            feature = layers.conv2d(
                feature, self.generation_unit * 8, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.conv2d_transpose(
                feature, self.generation_unit * 4, 3, padding="same", strides=2)
            feature = tf.nn.relu(feature)  # 12

            feature = layers.conv2d(
                feature, self.generation_unit * 4, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.conv2d_transpose(
                feature, self.generation_unit * 2, 4, padding="same", strides=2)
            feature = tf.nn.relu(feature)  # 24

            feature = layers.conv2d(
                feature, self.generation_unit * 2, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.conv2d_transpose(
                feature, self.generation_unit, 4, padding="same", strides=2)
            feature = tf.nn.relu(feature)  # 48

            feature = layers.conv2d(
                feature, self.generation_unit * 1, 3, padding="same")
            feature = tf.nn.relu(feature)
            feature = layers.conv2d_transpose(feature, 1, 3, padding="valid")
            if return_logits is True:
                return feature
            else:
                return tf.nn.sigmoid(feature)

    def _create_model(self):

        z_mean, z_log_var = self._inference_network(self.input_images)
        z_sample = self._sample_from_mvn(z_mean, tf.exp(z_log_var))
        x_recon = self._generation_network(z_sample, return_logits=True)

        KL = tf.reduce_mean(-tf.reduce_sum(
            -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
        NLL = tf.reduce_mean(
            tf.reduce_sum(
                -tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=layers.flatten(x_recon), labels=self.input_images),
                axis=1))
        ELBO = KL + NLL
        self.loss = -ELBO

        # # summary
        with tf.variable_scope("summaries"):
            # averaging between batch items
            # self.loss = tf.reduce_mean(self., name="loss")
            self.num_summaries.append(tf.summary.scalar("loss", self.loss))
            self.img_summaries.append(
                tf.summary.image(
                    "reconstruction",
                    tf.concat([self.input_as_image,
                               tf.nn.sigmoid(x_recon)], axis=2),
                    max_outputs=self.num_summary_images,
                ))

        if self.train:
            with tf.variable_scope("training"):
                # optimizer to minimize the loss function
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                grads, variables = zip(*optimizer.compute_gradients(self.loss))

                if self.gradient_clipping_norm is not None:
                    # gradient clipping by global norm, if required
                    grads = tf.clip_by_global_norm(grads,
                                                   self.gradient_clipping_norm)[0]

                grads_and_vars = list(zip(grads, variables))

                # training step operation
                self.training = optimizer.apply_gradients(
                    grads_and_vars, global_step=self.global_step)

    def _create_generation(self):
        z_sample = self._sample_from_mvn(
            tf.zeros([self.generation_batch_size, self.latent_unit]),
            tf.ones([self.generation_batch_size, self.latent_unit]))
        return z_sample
