import tensorflow as tf


def concrete_binary_sample(log_odds, temperature, hard=False, eps=10e-10):
    count = tf.shape(log_odds)[0]

    u = tf.random_uniform([count], minval=0, maxval=1)
    noise = tf.log(u + eps) - tf.log(1.0 - u + eps)

    y = log_odds + noise
    sig_y = tf.nn.sigmoid(y / temperature)

    if hard:
        sig_y_hard = tf.round(sig_y)
        sig_y = tf.stop_gradient(sig_y_hard - sig_y) + sig_y

    return y, sig_y


def concrete_binary_pre_sigmoid_sample(log_odds, temperature, eps=10e-10):
    count = tf.shape(log_odds)[0]

    u = tf.random_uniform([count], minval=0, maxval=1)
    noise = tf.log(u + eps) - tf.log(1.0 - u + eps)
    y = (log_odds + noise) / temperature

    return y


def concrete_binary_kl_mc_sample(
        y,
        prior_log_odds,
        prior_temperature,
        posterior_log_odds,
        posterior_temperature,
        eps=10e-10,
):

    # y_times_prior_temp = y * prior_temperature
    # log_prior = tf.log(prior_temperature + eps) - y_times_prior_temp + prior_log_odds - \
    #     2.0 * tf.log(1.0 + tf.exp(-y_times_prior_temp + prior_log_odds) + eps)

    # y_times_posterior_temp = y * posterior_temperature
    # log_posterior = tf.log(posterior_temperature + eps) - y_times_posterior_temp + posterior_log_odds - \
    #     2.0 * tf.log(1.0 + tf.exp(-y_times_posterior_temp + posterior_log_odds) + eps)

    # return log_posterior - log_prior

    logits = tf.stack([tf.zeros_like(y), -y * prior_temperature + prior_log_odds])
    log_sum_exp_prior = tf.reduce_logsumexp(logits, axis=0)
    log_prior = (tf.log(prior_temperature + eps) - y * (prior_temperature + 1) +
                 prior_log_odds - 2.0 * log_sum_exp_prior)

    logits_posterior = tf.stack([
        tf.zeros_like(posterior_log_odds),
        -y * posterior_temperature + posterior_log_odds
    ])

    log_sum_exp_posterior = tf.reduce_logsumexp(logits_posterior, axis=0)
    log_posterior = (
        tf.log(posterior_temperature + eps) - y * (posterior_temperature + 1) +
        posterior_log_odds - 2.0 * log_sum_exp_posterior)

    return log_posterior - log_prior
