import ssl
import certifi
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from numpy.random import default_rng

#Import the required libraries
# Fix HTTPS certificate verification before importing torchvision
ssl._create_default_https_context = (
    lambda: ssl.create_default_context(cafile=certifi.where())
)

# Download MNIST datasets
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
)

# Read raw MNIST binary files
def load_dataset(path, is_label=False):
    offset = 8 if is_label else 16

    with open(path, "rb") as file:
        data = np.frombuffer(
            file.read(),
            dtype=np.uint8,
            offset=offset,
        )

    return data


# File paths
data_dir = "./data/MNIST/raw"

X_train = load_dataset(
    f"{data_dir}/train-images-idx3-ubyte"
)

y_train = load_dataset(
    f"{data_dir}/train-labels-idx1-ubyte",
    is_label=True,
)

X_test = load_dataset(
    f"{data_dir}/t10k-images-idx3-ubyte"
)

y_test = load_dataset(
    f"{data_dir}/t10k-labels-idx1-ubyte",
    is_label=True,
)

# Reshape images: 28 x 28 = 784 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)


'''# Check the results
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Visualize some samples from the training dataset
fig, axes = plt.subplots(4, 10, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    ax.imshow(image, cmap="gray")
    ax.set_title(f"label = {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()'''


# Convert pixels from 0-255 to 0.0-1.0, avoiding overflow when multiplying by weights
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# One-hot encode labels
identity_matrix = np.eye(10)

y_train = identity_matrix[y_train]
y_test = identity_matrix[y_test]


#Some useful functions for the neural network
def relu(x):
    return np.maximum(0, x)

def batch_softmax(x):
    row_max = np.max(x, axis = 1, keepdims = True)
    exp_x = np.exp(x - row_max)
    return exp_x / np.sum(exp_x, axis = 1, keepdims = True)

def batch_cross_entropy(y, y_hat): 
    total_loss = -np.sum(y * np.log(y_hat + 1e-15), axis=1, keepdims=True) #prevent log 0
    return np.mean(total_loss)

def diff_relu(x):
    return (x > 0).astype(float)


def augment_batch(X_batch, augmentation_rng, max_shift = 2):
    """Shift each 28x28 image independently; fill uncovered pixels with zero."""
    images = X_batch.reshape(-1, 28, 28)
    shifted = np.zeros_like(images)
    offsets = augmentation_rng.integers(-max_shift, max_shift + 1,
                                        size=(len(images), 2))
    for image_index, (dy, dx) in enumerate(offsets):
        # Copy overlapping pixels; do not wrap edge pixels to the opposite side.
        src_y = slice(max(0, -dy), min(28, 28 - dy))
        src_x = slice(max(0, -dx), min(28, 28 - dx))
        dst_y = slice(max(0, dy), min(28, 28 + dy))
        dst_x = slice(max(0, dx), min(28, 28 + dx))
        shifted[image_index, dst_y, dst_x] = images[image_index, src_y, src_x]
    return shifted.reshape(X_batch.shape)


#Initialize weights and biases
rng = default_rng(seed = 42)
limit = np.sqrt(6 / 784)
w1 = rng.uniform(low = -limit, high = limit, size = (784, 16)) #Mind the defference between the regression and the NN, because Regression only passes one input, while NN passes a batch of inputs, so after one Node, the result is still a matix rather than a value.
limit = np.sqrt(6 / 16)
w2 = rng.uniform(low=-limit, high=limit, size=(16, 16))
w3 = rng.uniform(low=-limit, high=limit, size=(16, 10))

b1 = np.zeros((1, 16))
b2 = np.zeros((1, 16))
b3 = np.zeros((1, 10))


#Train-validation split
split = int(0.95 * len(X_train))
shuffle = rng.permutation(len(X_train))
X_train, y_train = X_train[shuffle], y_train[shuffle]
X_train_split = X_train[:split]
y_train_split = y_train[:split]
X_val_split = X_train[split:]
y_val_split = y_train[split:]

#Iteration parameters
epochs = 1000
batch_size = 32
max_shift = 2  # Random translation by up to 2 pixels; set to 0 to disable.
augmentation_rng = default_rng(seed = 42)  # Separate from weights/shuffling/dropout.

# Early stopping parameters
patience  = 18
minidelta = 0.0001
patience_counter = 0

#checkpoint parameters
best_val_cost = np.inf
best_w1, best_w2, best_w3 = None, None, None
best_b1, best_b2, best_b3 = None, None, None

#lr reduction parameters
lr = 0.05 
min_lr = 1e-6
lr_reduction_factor = 0.5
lr_maxcon_reductions = 8 
lr_reduction_counter = 0

#L2 regularization parameters
lambda_ = 0.15

#Dropout parameters
pkeep = 0.9

#Tracking validation and training cost and accuracy
batches = (len(X_train_split) + batch_size - 1) // batch_size
val_cost_list = []
val_accuracy = []
train_cost_list = []
train_accuracy = []


for i in range(epochs):
    shuffle = rng.permutation(len(X_train_split))
    X_train_split, y_train_split = X_train_split[shuffle], y_train_split[shuffle]
    batch_count = 0
    for j in range(0, len(X_train_split), batch_size):
        batch_count += 1
        end = j + batch_size
        X_train_batch = X_train_split[j:end]
        y_train_batch = y_train_split[j:end]
        X_train_batch = augment_batch(X_train_batch, augmentation_rng, max_shift)

        #Forward Pass
        z1 = np.dot(X_train_batch, w1) + b1
        a1 = relu(z1)

        a1_mask = rng.binomial(1, pkeep, size = a1.shape)
        a1_mask = a1_mask.astype(float)
        a1 *= a1_mask / pkeep

        z2 = np.dot(a1, w2) + b2
        a2 = relu(z2)

        a2_mask = rng.binomial(1, pkeep, size = a2.shape)
        a2_mask = a2_mask.astype(float)
        a2 *= a2_mask / pkeep

        z3 = np.dot(a2, w3) + b3
        a3 = batch_softmax(z3)

        cost = batch_cross_entropy(y_train_batch, a3)

        #Backward Pass
        delta3 = a3 - y_train_batch
        delta2 = np.dot(delta3, w3.T) * diff_relu(z2) * (a2_mask / pkeep)
        delta1 = np.dot(delta2, w2.T) * diff_relu(z1) * (a1_mask / pkeep)

        dw3 = np.dot(a2.T, delta3) / X_train_batch.shape[0]
        dw2 = np.dot(a1.T, delta2) / X_train_batch.shape[0]
        dw1 = np.dot(X_train_batch.T, delta1) / X_train_batch.shape[0]
        

        db3 = np.sum(delta3, axis=0, keepdims=True) / X_train_batch.shape[0]
        db2 = np.sum(delta2, axis=0, keepdims=True) / X_train_batch.shape[0]
        db1 = np.sum(delta1, axis=0, keepdims=True) / X_train_batch.shape[0]    

        w1 -= (lr * (dw1 + (lambda_ * w1) / X_train_split.shape[0]))
        w2 -= (lr * (dw2 + (lambda_ * w2) / X_train_split.shape[0]))
        w3 -= (lr * (dw3 + (lambda_ * w3) / X_train_split.shape[0]))

        b1 -= lr * db1
        b2 -= lr * db2
        b3 -= lr * db3
    
    

    #Validation
    z1_val = np.dot(X_val_split, w1) + b1
    a1_val = relu(z1_val)
    z2_val = np.dot(a1_val, w2) + b2
    a2_val = relu(z2_val)
    z3_val = np.dot(a2_val, w3) + b3
    a3_val = batch_softmax(z3_val)

    val_cost = batch_cross_entropy(y_val_split, a3_val)
    val_cost_list.append(val_cost)
    
    predictions = np.argmax(a3_val, axis=1)
    labels = np.argmax(y_val_split, axis=1)
    accuracy = np.mean(predictions == labels)
    val_accuracy.append(accuracy)

    #Training
    z1_train = np.dot(X_train_split, w1) + b1
    a1_train = relu(z1_train)   
    z2_train = np.dot(a1_train, w2) + b2
    a2_train = relu(z2_train)
    z3_train = np.dot(a2_train, w3) + b3
    a3_train = batch_softmax(z3_train)

    train_cost = batch_cross_entropy(y_train_split, a3_train)
    train_cost_list.append(train_cost)
    train_predictions = np.argmax(a3_train, axis=1)
    train_labels = np.argmax(y_train_split, axis=1)
    train_acc = np.mean(train_predictions == train_labels)
    train_accuracy.append(train_acc)
    
    #Print progress every 100 epochs
    if (i + 1) % 100 == 0:
        print(f"Epoch {i + 1}/{epochs}, Validation Cost: {val_cost:.4f}, Training Cost: {train_cost:.4f}")

    # Early Stopping & lr reduction
    if val_cost < best_val_cost:
        best_w1, best_w2, best_w3 = np.copy(w1), np.copy(w2), np.copy(w3)
        best_b1, best_b2, best_b3 = np.copy(b1), np.copy(b2), np.copy(b3)

    if best_val_cost - val_cost > minidelta:
        best_val_cost = val_cost
        patience_counter = 0
        lr_reduction_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            if lr * lr_reduction_factor >= min_lr and lr_reduction_counter < lr_maxcon_reductions:
                lr *= lr_reduction_factor
                print(f"Epoch {i + 1}: Reducing learning rate to {lr:.6f}")
                patience_counter = 0
                lr_reduction_counter += 1
            else:
                print(f"Epoch {i + 1}: Early stopping triggered.")
                break



test_z1 = np.dot(X_test, best_w1) + best_b1
test_a1 = relu(test_z1)
test_z2 = np.dot(test_a1, best_w2) + best_b2
test_a2 = relu(test_z2)
test_z3 = np.dot(test_a2, best_w3) + best_b3
test_a3 = batch_softmax(test_z3)

test_predictions = np.argmax(test_a3, axis=1)
test_labels = np.argmax(y_test, axis=1)
test_accuracy = np.mean(np.equal(test_predictions, test_labels))
print(f"Test Accuracy: {test_accuracy:.4f}")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Cost graph
epoch_numbers = range(1, len(val_cost_list) + 1)
ax1.plot(epoch_numbers, val_cost_list, marker="o", color="tab:green", label="Validation")
ax1.plot(epoch_numbers, train_cost_list, marker="s", color="tab:pink", label="Training")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cost")
ax1.set_title("Cost")
ax1.grid(True)
ax1.legend()

# Accuracy graph
ax2.plot(epoch_numbers, val_accuracy, marker="o", color="tab:green", label="Validation")
ax2.plot(epoch_numbers, train_accuracy, marker="s", color="tab:pink", label="Training")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy")
ax2.grid(True)
ax2.legend()


plt.tight_layout()
plt.show()
