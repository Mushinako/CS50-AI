# Traffic

## Neural Network Experimentation Process

The first things I tried is to make a complicated network:

```py
# Input 1×30×30×3
model.add(tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
# 1×30×30×3 => 16×28×28×3 (3×3 Conv2D)
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
# 16×28×28×3 => 16×14×14×3 (2×2 MaxPool2D)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
# 16×14×14×3 => 36×12×12×3 (3×3 Conv2D)
model.add(tf.keras.layers.Conv2D(36, (3, 3), activation="relu"))
# 36×12×12×3 => 36×6×6×3 (2×2 MaxPool2D)
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
# 36×6×6×3 => 3888 (Flatten)
model.add(tf.keras.layers.Flatten())
# 3888 => 1440 (Linear)
model.add(tf.keras.layers.Dense(1440, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
# 1440 => 500 (Linear)
model.add(tf.keras.layers.Dense(500, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
# 500 => 160 (Linear)
model.add(tf.keras.layers.Dense(160, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
# 160 => 43 (Linear)
model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
```

There a few problems with this network:

* The model is overcomplicated, with too many layers and nodes making it extremely difficult to find the minimum loss.
* The dropout rate may have been too high compared to the training size, such that many nodes may not be fully trained.

Addressing these issues, the model is improved to such:

```py
# Input 1×30×30×3
model.add(tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
# 1×30×30×3 => 36×27×27×3 (4×4 Conv2D)
model.add(tf.keras.layers.Conv2D(36, (4, 4), activation="relu"))
# 36×27×27×3 => 36×9×9×3 (3×3 MaxPool2D)
model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
# 36×9×9×3 => 8748 (Flatten)
model.add(tf.keras.layers.Flatten())
# 8748 => 160 (Linear)
model.add(tf.keras.layers.Dense(160, activation="relu"))
model.add(tf.keras.layers.Dropout(0.25))
# 160 => 43
model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
```
