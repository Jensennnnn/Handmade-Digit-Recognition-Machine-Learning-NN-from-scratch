import torch 
import torchvision
import ssl
import certifi
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

class DNN:
    def __init__(self, seed=42):
        self.layers = []
        self.weights = []
        self.biases = []
        self.zs = []
        self.acts = []

        self.pkeeps = []
        self.masks = []
        self.weight_decays = []
        self.rng = default_rng(seed)
        self.split_rng = default_rng(seed + 1)
        self.shuffle_rng = default_rng(seed + 2)
        self.augmentation_rng = default_rng(seed + 3)
        self.inputs = None
        self.training_size = None

        self.val_costs = []
        self.val_accuracies = []

        self.train_costs = []
        self.train_accuracies = []

        self.early_stop_min_delta = 0.0
        self.early_stop_patience = 5
        self.early_stopping_enabled = False

        self.adaptive_lr_enabled = False
        self.adaptive_lr_maxcon_reductions = 4
        self.adaptive_lr_reduction_factor = 0.5
        self.min_adaptive_lr = 1e-6

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        m = np.max(x, axis = 1, keepdims = True)
        exp_x = np.exp(x - m)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy(self, y, y_hat):
        total_loss = -np.sum(y * np.log(y_hat + 1e-15), axis = 1, keepdims = True)
        return np.mean(total_loss)

    def diff_relu(self, x):
        return (x > 0).astype(float)

    def create_layer_params(self, l2_reg_rate = 0.0):
        if len(self.layers) > 1:
            last_layer = self.layers[-1]
            prev_layer = self.layers[-2]
            limit = np.sqrt(6 / prev_layer)
            w = self.rng.uniform(-limit, limit, (prev_layer, last_layer))
            self.weights.append(w)
            b = np.zeros((1, last_layer))
            self.biases.append(b)
            self.weight_decays.append(l2_reg_rate)

    def add_layer(self, neurons, l2_reg_rate = 0.00, pkeep = 1.00):
        self.layers.append(neurons)
        self.create_layer_params(l2_reg_rate)
        self.pkeeps.append(pkeep)
    
    def forward_pass(self, X, Y = None, params = None, inference = False):
        self.acts.clear()
        self.zs.clear()
        self.masks.clear()

        weights, biases = (self.weights, self.biases) if params is None else params
        if not inference:
            input_mask = self.rng.binomial(n = 1, p = self.pkeeps[0], size = X.shape)
            X = X * input_mask / self.pkeeps[0]
            self.masks.append(input_mask)
        self.inputs = X

        for i in range(len(self.weights)):
            if i == 0:
                z = np.dot(X, weights[0]) + biases[0]
            else:
                z = np.dot(self.acts[-1], weights[i]) + biases[i]

            if i != len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.softmax(z)

            if not inference:
                mask = np.ones_like(a)
                if i != len(weights) - 1:
                    mask = self.rng.binomial(n = 1, p = self.pkeeps[i + 1], size = a.shape)
                    a *= mask / self.pkeeps[i + 1]
                self.masks.append(mask)

            self.acts.append(a)
            self.zs.append(z)

        if Y is not None:
            cost_raw = self.cross_entropy(Y, self.acts[-1])
            squared_weights = 0
            if not inference:
                for i in range(len(self.weights)):
                    squared_weights += self.weight_decays[i] * np.sum(weights[i] ** 2)
                N = self.training_size if self.training_size is not None else len(X)
                l2_reg_cost = cost_raw + squared_weights / (2 * N)
                return self.acts[-1], cost_raw, l2_reg_cost
            else:
                return self.acts[-1], cost_raw
        else:
            return self.acts[-1]

    def backward_pass(self, X, Y, lr):
        deltas = [None] * len(self.weights)
        dw = [None] * len(self.weights)
        db = [None] * len(self.weights)

        for i in range(len(self.weights)-1, -1, -1):
            if i == len(self.weights) - 1:
                delta = self.acts[-1] - Y
            
            else:
                delta = np.dot(deltas[i+1], self.weights[i+1].T) * self.diff_relu(self.zs[i])
            
            if i != len(self.weights) - 1:
                delta *= self.masks[i+1] / self.pkeeps[i+1]
            deltas[i] = delta
        
        N = self.training_size if self.training_size is not None else len(X)
        for i in range(len(self.weights)-1, -1, -1):
            if i == 0:
                dw[i] = np.dot(self.inputs.T, deltas[i]) / X.shape[0] + (self.weight_decays[i] / N) * self.weights[i]
            else:
                dw[i] = np.dot(self.acts[i-1].T, deltas[i]) / X.shape[0] + (self.weight_decays[i] / N) * self.weights[i]
            
            db[i] = np.mean(deltas[i], axis = 0, keepdims = True)

        for i in range(len(self.weights)-1, -1, -1):
            self.weights[i] -= lr * dw[i]
            self.biases[i] -= lr * db[i]
        return dw, db
    
    def augment_batch(self, X_batch, augmentation_rng, max_shift = 2):
        images = X_batch.reshape(-1, 28, 28)
        shifted = np.zeros_like(images)
        offsets = augmentation_rng.integers(-max_shift, max_shift + 1, size = (len(images), 2))
        for image_index, (dy, dx) in enumerate(offsets):
            src_y = slice(max(0, -dy), min(28, 28 - dy))
            src_x = slice(max(0, -dx), min(28, 28 - dx))
            dst_y = slice(max(0, dy), min(28, 28 + dy))
            dst_x = slice(max(0, dx), min(28, 28 + dx))
            shifted[image_index, dst_y, dst_x] = images[image_index, src_y, src_x]
        return shifted.reshape(X_batch.shape)

    def conf_early_stopping(self, enabled = False, early_stop_patience = 5, min_delta = 0.00):
        self.early_stopping_enabled = enabled
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = min_delta
    
    def conf_adaptive_lr(self, enabled = False, min_lr = 1e-6, lr_reduction_factor = 0.5, lr_maxcon_reductions = 4):
        self.adaptive_lr_enabled = enabled
        self.min_adaptive_lr = min_lr
        self.adaptive_lr_reduction_factor = lr_reduction_factor
        self.adaptive_lr_maxcon_reductions = lr_maxcon_reductions

    def train(self, X, Y, train_split, initial_lr, batch_size = None, epochs = 1,
              max_shift = 2, save_path = "model.npz"):
        self.val_costs.clear()
        self.val_accuracies.clear()
        self.train_costs.clear()
        self.train_accuracies.clear()

        shuffle = self.split_rng.permutation(len(X))
        X_train, Y_train = X[shuffle], Y[shuffle]
        train_split = int(len(X) * train_split)
        X_train_split, Y_train_split = X_train[:train_split], Y_train[:train_split]
        X_val_split, Y_val_split = X_train[train_split:], Y_train[train_split:]
        self.training_size = len(X_train_split)
        self.best_epoch = None
        self.lr_history = []

        best_check_val_cost = np.inf
        bestWs = []
        bestBs = []

        patience_counter = 0
        best_stop_val_cost = np.inf

        lr = initial_lr
        lr_reduction_counter = 0

        if batch_size is None:
            batch_size = X_train_split.shape[0]
        
        batches = (len(X_train_split) + batch_size - 1) // batch_size

        print("Total batches = ", batches)
        print("#---------------------------------TRAINING STARTED---------------------------------#")
        for j in range(epochs):
            print(f"#---------------------Epoch: {j+1}/{epochs}---------------------#")
            self.lr_history.append(lr)
            shuffle = self.shuffle_rng.permutation(len(X_train_split))
            X_train_split, Y_train_split = X_train_split[shuffle], Y_train_split[shuffle]

            batch_counter = 0
            for i in range(0, len(X_train_split), batch_size):
                end = min(i + batch_size, len(X_train_split))
                batch_counter += 1
                X_train_batch = X_train_split[i:end]
                Y_train_batch = Y_train_split[i:end]
                if max_shift:
                    X_train_batch = self.augment_batch(X_train_batch, self.augmentation_rng, max_shift)

                self.forward_pass(X_train_batch, Y_train_batch)
                self.backward_pass(X_train_batch, Y_train_batch, lr)
            
            predictions, val_cost = self.forward_pass(X_val_split, Y_val_split, inference = True)
            
            prediction = np.argmax(predictions, axis=1)
            actual = np.argmax(Y_val_split, axis=1)
            equals = np.equal(prediction, actual)
            accuracy = np.mean(equals) * 100
            print("Validation cost: ", val_cost)
            print("Validation Accuracy: ", accuracy)
            self.val_costs.append(float(val_cost))
            self.val_accuracies.append(float(accuracy))

            predictions, train_cost = self.forward_pass(X_train_split, Y_train_split, inference = True)
            prediction = np.argmax(predictions, axis=1)
            actual = np.argmax(Y_train_split, axis=1)
            equals = np.equal(prediction, actual)
            accuracy = np.mean(equals) * 100
            print("Training cost: ", train_cost)
            print("Training Accuracy: ", accuracy)
            self.train_costs.append(float(train_cost))
            self.train_accuracies.append(float(accuracy))

            if val_cost < best_check_val_cost:
                best_check_val_cost = val_cost
                self.best_epoch = j + 1
                bestWs.clear()
                bestBs.clear()
                for i in range(len(self.weights)):
                    bestWs.append(np.copy(self.weights[i]))
                    bestBs.append(np.copy(self.biases[i]))
                print("Update epoch:", j + 1)

            if self.early_stopping_enabled or self.adaptive_lr_enabled:
                if val_cost < best_stop_val_cost - self.early_stop_min_delta:
                    best_stop_val_cost = val_cost
                    patience_counter = 0
                    lr_reduction_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        if (self.adaptive_lr_enabled
                                and lr_reduction_counter < self.adaptive_lr_maxcon_reductions
                                and lr > self.min_adaptive_lr):
                            lr = max(self.min_adaptive_lr, lr * self.adaptive_lr_reduction_factor)
                            print("Reduced LR:", lr)
                            lr_reduction_counter += 1
                            patience_counter = 0
                        else:
                            if self.early_stopping_enabled:
                                print("Early stopping at epoch:", j + 1)
                                break
            
        weights_obj = np.empty(len(bestWs), dtype=object)
        weights_obj[:] = bestWs

        biases_obj = np.empty(len(bestBs), dtype=object)
        biases_obj[:] = bestBs

        np.savez(save_path, weights = weights_obj, biases = biases_obj)

        return [bestWs, bestBs]

    def plot(self):
        epochs = range(1, len(self.val_costs) + 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax1.set_xlim(1, max(2, len(self.val_costs)))
        ax2.set_xlim(1, max(2, len(self.val_costs)))

        # Cost graph
        ax1.plot(epochs, self.val_costs, marker="o", color="tab:red", label="Val Cost")
        ax1.plot(epochs, self.train_costs, marker="o", color="tab:blue", label="Train Cost")
        ax1.set_ylabel("Validation/Training Cost")
        ax1.set_title("Validation/Training Cost")
        ax1.legend()
        ax1.grid(True)

        # Accuracy graph
        ax2.plot(epochs, self.val_accuracies, marker="o", color="tab:orange", label="Val Accuracy")
        ax2.plot(epochs, self.train_accuracies, marker="o", color="tab:green", label="Train Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation/Training Accuracy (%)")
        ax2.set_title("Validation/Training Accuracy ")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()




def main():
    data_root = Path(__file__).resolve().parent.parent / "data"

    ssl._create_default_https_context = (
        lambda: ssl.create_default_context(cafile=certifi.where())
    )

    train_dataset = torchvision.datasets.MNIST(
        root=str(data_root),
        train=True,
        download=True,
    )

    test_dataset = torchvision.datasets.MNIST(
        root=str(data_root),
        train=False,
        download=True,
    )

    def load_dataset(path, is_label=False):
        offset = 8 if is_label else 16

        with open(path, "rb") as file:
            data = np.frombuffer(
                file.read(),
                dtype=np.uint8,
                offset=offset,
            )

        return data

    data_dir = data_root / "MNIST" / "raw"

    X_train = load_dataset(
        f"{data_dir}/train-images-idx3-ubyte"
    )
    Y_train = load_dataset(
        f"{data_dir}/train-labels-idx1-ubyte",
        is_label=True,
    )
    X_test = load_dataset(
        f"{data_dir}/t10k-images-idx3-ubyte"
    )
    Y_test = load_dataset(
        f"{data_dir}/t10k-labels-idx1-ubyte",
        is_label=True,
    )


    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    eye = np.identity(10)
    Y_train = eye[Y_train]
    Y_test = eye[Y_test]

    model = DNN()
    model.add_layer(784) 
    model.add_layer(256, l2_reg_rate=5e-4, pkeep=0.8) 
    model.add_layer(128, l2_reg_rate=5e-4, pkeep=0.8) 
    model.add_layer(64, l2_reg_rate=5e-4, pkeep=0.9) 
    model.add_layer(10)

    model.conf_early_stopping(True, early_stop_patience = 10, min_delta = 5e-4) 
    model.conf_adaptive_lr(True,  min_lr = 1e-5, lr_reduction_factor = 0.5, lr_maxcon_reductions = 6) 

    best_params = model.train(X_train, Y_train, 0.9, 0.01, 64, 1000) 

    preds, cost = model.forward_pass(X_test, Y_test, best_params, inference=True)
    predictions = np.argmax(preds, axis=1)
    actual = np.argmax(Y_test, axis=1)
    equals = np.equal(predictions, actual)
    accuracy = np.mean(equals) * 100

    print(f'Test Cost = {cost}\nTest Accuracy = {accuracy:.3f}%')
    model.plot()


if __name__ == "__main__":
    main()
