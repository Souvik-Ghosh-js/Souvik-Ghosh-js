import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load your original h5 model
loaded_model = tf.keras.models.load_model(r'C:\Users\91629\Desktop\Aquafaze\h2\app\models\Gener_detect.h5')

# Quantization
quantized_model = tfmot.quantization.keras.quantize_model(loaded_model)

# Pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=0,
        end_step= 18
    )
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(quantized_model, **pruning_params)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the pruned model and save it
# pruned_model.fit(x_train, y_train, epochs=epochs)
pruned_model.save('reduced_model.h5')