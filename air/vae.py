import tensorflow as tf
import tensorflow.contrib.layers as layers


def vae(inputs,
        input_dim,
        rec_hidden_units,
        latent_dim,
        gen_hidden_units,
        likelihood_std=0.0,
        activation=tf.nn.softplus):

    input_size = tf.shape(inputs)[0]

    next_layer = inputs
    for i in range(len(rec_hidden_units)):
        with tf.variable_scope("recognition_" + str(i + 1)) as scope:
            next_layer = layers.fully_connected(
                next_layer, rec_hidden_units[i], activation_fn=activation, scope=scope)

    with tf.variable_scope("rec_mean") as scope:
        recognition_mean = layers.fully_connected(
            next_layer, latent_dim, activation_fn=None, scope=scope)
    with tf.variable_scope("rec_log_variance") as scope:
        recognition_log_variance = layers.fully_connected(
            next_layer, latent_dim, activation_fn=None, scope=scope)

    with tf.variable_scope("rec_sample"):
        standard_normal_sample = tf.random_normal([input_size, latent_dim])
        recognition_sample = recognition_mean + standard_normal_sample * tf.sqrt(
            tf.exp(recognition_log_variance))

    next_layer = recognition_sample
    for i in range(len(gen_hidden_units)):
        with tf.variable_scope("generative_" + str(i + 1)) as scope:
            next_layer = layers.fully_connected(
                next_layer, gen_hidden_units[i], activation_fn=activation, scope=scope)

    with tf.variable_scope("gen_mean") as scope:
        generative_mean = layers.fully_connected(
            next_layer, input_dim, activation_fn=None, scope=scope)

    with tf.variable_scope("gen_sample"):
        standard_normal_sample2 = tf.random_normal([input_size, input_dim])
        generative_sample = generative_mean + standard_normal_sample2 * likelihood_std
        reconstruction = tf.nn.sigmoid(generative_sample)

    return reconstruction, recognition_mean, recognition_log_variance, recognition_sample


def vae_generation(batch_size,
                   input_dim,
                   latent_dim,
                   latent_prior_mean,
                   latent_prior_log_var,
                   gen_hidden_units,
                   likelihood_std=0.0,
                   activation=tf.nn.softplus,
                   sample_from_mean=False):

    input_size = batch_size

    with tf.variable_scope("rec_sample"):
        standard_normal_sample = tf.random_normal([input_size, latent_dim])
        latent_sample = latent_prior_mean + standard_normal_sample * tf.sqrt(
            tf.exp(latent_prior_log_var))

    next_layer = latent_sample
    for i in range(len(gen_hidden_units)):
        with tf.variable_scope("generative_" + str(i + 1)) as scope:
            next_layer = layers.fully_connected(
                next_layer, gen_hidden_units[i], activation_fn=activation, scope=scope)

    with tf.variable_scope("gen_mean") as scope:
        generative_mean = layers.fully_connected(
            next_layer, input_dim, activation_fn=None, scope=scope)

    with tf.variable_scope("gen_sample"):
        standard_normal_sample2 = tf.random_normal([input_size, input_dim])
        generative_sample = generative_mean + standard_normal_sample2 * likelihood_std
        samples = tf.nn.sigmoid(generative_sample)

    if sample_from_mean is True:
        samples = tf.nn.relu(tf.sign(samples - tf.random_uniform(tf.shape(samples))))

    return latent_sample, samples
