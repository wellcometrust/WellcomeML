from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf

class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs):
        X_val = self.validation_data[0]
        Y_val = self.validation_data[1]

        Y_pred = self.model.predict(X_val) > 0.5
        f1 = round(f1_score(Y_val, Y_pred, average='micro'), 4)
        p = round(precision_score(Y_val, Y_pred, average='micro'), 4)
        r = round(recall_score(Y_val, Y_pred, average='micro'), 4)
        print(f" - val metrics: P {p:.4f} R {r:.4f} F {f1:.4f}")
        return
