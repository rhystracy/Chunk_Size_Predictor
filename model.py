import numpy as np
import tensorflow as tf
import keras

class TRANS_Model():
    def __init__(self, shape):
        self.model = self.build_trans_model(
            input_shape=shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=[tf.metrics.MeanAbsoluteError()])
    
    def trans_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0): #normalize, attention, normalize, feedforward, normalize
        # Normalization and Attention
        #mask = np.ones((inputs.shape[1],1))
        #mask[56:] = 0
        #inputs = tf.boolean_mask(inputs, mask, axis=None, name='boolean_mask')
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        
        normalized_out = x + res
        return normalized_out

    def build_trans_model(
        self,
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
    ):
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.trans_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = tf.keras.layers.Dense(dim, activation="relu")(x)
            x = tf.keras.layers.Dropout(mlp_dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs, outputs)
    
    def train(self, train_x, train_y):
        callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

        self.model.fit(
            train_x,
            train_y,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )
    
    def predict(self, test_x):
        pred = self.model.predict(x=test_x)
        return pred
    
    def evaluate(self, eval_x, eval_y):
        evaluation = self.model.evaluate(eval_x, eval_y, verbose=0)
        return evaluation

