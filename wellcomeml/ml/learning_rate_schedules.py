import tensorflow as tf


class ExponentialDecayWithWarmRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, steps_per_epoch, learning_rate_decay, epochs):
        super(ExponentialDecayWithWarmRestarts, self).__init__()
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

    def __call__(self, step):
        step_in_epoch = step % self.steps_per_epoch
        decay = self.learning_rate_decay ** (20 * step_in_epoch / self.steps_per_epoch)
        return self.learning_rate * decay
