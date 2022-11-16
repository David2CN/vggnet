
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from tensorflow.keras.utils import image_dataset_from_directory


def get_dataset(data_dir: str, batch_size: int=16, 
            image_size: Tuple[int, int]=(224, 224), shuffle: bool=True):

    data = image_dataset_from_directory(data_dir, 
                                        image_size=image_size,
                                        batch_size=batch_size, 
                                        shuffle=shuffle)
    return data
    

def apply_gradient(optimizer: tf.keras.optimizers.Optimizer, loss_object: tf.keras.losses.Loss , 
                    model: tf.keras.models.Model, x: tf.Tensor, y: tf.Tensor):
    '''
    applies the gradients to the trainable model weights
    '''
    
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_object(y, logits)
  
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return logits, loss


def epoch_train(train_dataset: tf.data.Dataset, optimizer: tf.keras.optimizers.Optimizer, 
                loss_object: tf.keras.losses.Loss, model: tf.keras.models.Model):
    '''
    Computes the train loss then updates the weights and metrics for one epoch.
    '''
    losses = []
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # Iterate through all batches of training data
    for x, y in tqdm(train_dataset):

        logits, loss_vals = apply_gradient(optimizer, loss_object, model, x, y)
        losses.extend(loss_vals)

        train_acc_metric.update_state(y, logits)

    accuracy = train_acc_metric.result().numpy()
    loss = tf.reduce_mean(losses).numpy()
    return loss, accuracy


def epoch_val(val_dataset: tf.data.Dataset, loss_object: tf.keras.losses.Loss, 
                model: tf.keras.models.Model):
    '''
    Computes the validation loss and metrics for one epoch.
    '''
    losses = []
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in tqdm(val_dataset):

        logits = model(x)
        loss_vals = loss_object(y, logits)
        losses.extend(loss_vals)

        val_acc_metric.update_state(y, logits)

    accuracy = val_acc_metric.result().numpy()
    loss = tf.reduce_mean(losses).numpy()
    return loss, accuracy


def train(model: tf.keras.models.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
          optimizer: tf.keras.optimizers.Optimizer, loss_object: tf.keras.losses.Loss,epochs: int=10,):

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(1, epochs+1):
        train_loss, train_acc = epoch_train(train_dataset, optimizer, loss_object, model)
        val_loss, val_acc = epoch_val(val_dataset, loss_object, model)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)


        print(f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} val_loss: {val_loss:.4f}, accuracy: {train_acc:.4f} val_accuracy: {val_acc:.4f}")

    train_losses, val_losses = tf.convert_to_tensor(train_losses).numpy(), tf.convert_to_tensor(val_losses).numpy()
    train_accs, val_accs = tf.convert_to_tensor(train_accs).numpy(), tf.convert_to_tensor(val_accs).numpy()
    
    result = {
        "loss": {"train": train_losses, "val": val_losses},
        "acc": {"train": train_accs, "val": val_accs}
    }
    
    return result


def predict(val_dataset: tf.data.Dataset, model: tf.keras.models.Model):
    '''
    predicts on a given dataset
    '''
    targets = []
    outputs = []

    for x, y in tqdm(val_dataset):

        logits = model(x)
        targets.extend(y)
        outputs.extend(logits)

    preds = tf.argmax(tf.convert_to_tensor(outputs), axis=1)
    actual = tf.convert_to_tensor(targets, dtype="int64")
    
    return actual.numpy(), preds.numpy()


def plot_metrics(results: dict):

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    losses = results["loss"]
    accs = results["acc"]

    x = range(1, len(losses["val"])+1)

    ax.plot(x, losses["train"], label="train")
    ax.plot(x, losses["val"], label="validation")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.set_xticks(x)
    ax.set_xlim(1, x[-1])

    ax2.plot(x, accs["train"], label="train")
    ax2.plot(x, accs["val"], label="validation")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_xticks(x)
    ax2.set_xlim(1, x[-1])
    
    ax.legend()
    ax2.legend()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list=[0,1,2,3], title: str=''):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax,
                                            xticks_rotation=45, cmap="Blues")

    if title == '':
        title = f"Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%"

    plt.title(title)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
  
    plt.show()


if __name__=="__main__":
    a = [0, 1, 1, 0, 1, 2, 3, 2]
    b = [0, 1, 1, 0, 0, 2, 3, 3]

    classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

    plot_confusion_matrix(a, b, labels=classes)
