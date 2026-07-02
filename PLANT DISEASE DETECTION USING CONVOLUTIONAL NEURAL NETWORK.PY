# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from docx import Document

# Dataset path
data_dir = "Dataset"

# Image preprocessing with validation split
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Training data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build CNN model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(train_generator.num_classes, activation='softmax'))

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

# Evaluate
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# ===============================
# 📊 SAVE TO EXCEL
# ===============================

history_df = pd.DataFrame(history.history)

# Add final results
results_df = pd.DataFrame({
    "Metric": ["Validation Loss", "Validation Accuracy"],
    "Value": [loss, accuracy]
})

# Save to Excel
with pd.ExcelWriter("training_results.xlsx") as writer:
    history_df.to_excel(writer, sheet_name="Training History", index=False)
    results_df.to_excel(writer, sheet_name="Final Results", index=False)

# ===============================
# 📝 SAVE TO WORD DOCUMENT
# ===============================

doc = Document()
doc.add_heading('Plant Disease CNN Training Report', 0)

# Add results
doc.add_heading('Final Results', level=1)
doc.add_paragraph(f"Validation Loss: {loss:.4f}")
doc.add_paragraph(f"Validation Accuracy: {accuracy*100:.2f}%")

# Add class labels
doc.add_heading('Class Labels', level=1)
for label, index in train_generator.class_indices.items():
    doc.add_paragraph(f"{index} : {label}")

# Add training history summary
doc.add_heading('Training Summary', level=1)
doc.add_paragraph(str(history_df.describe()))

# Save Word file
doc.save("training_report.docx")

# ===============================
# 📈 PLOT & SAVE FIGURE
# ===============================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.savefig("training_plot.png")
plt.show()

# Save model
model.save("plant_disease_model.h5")

