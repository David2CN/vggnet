
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model


class VGGBlock(Model):
    """
    custom conv2d layer coupled with relu activation and max pooling
    """
    def __init__(self,  filters: int, kernel_size: int, repetitions: int, 
                    pool_size: int=2, strides: int=2, block: int=0, dropout: float=None):
        super(VGGBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions
        
        if dropout:
            self.dropout = Dropout(dropout)

        layers = []
        for i in range(repetitions):
            layers.append(Conv2D(filters, kernel_size, strides=1,
                                padding="same", activation="relu", name=f"conv{block}_{i+1}"))
        
        self.rows = tf.keras.Sequential(layers)
        self.pool = MaxPooling2D(pool_size=pool_size, strides=strides, name=f"pool{block}")

    
    def call(self, inputs: tf.Tensor):
        x = self.rows(inputs)
        out = self.pool(x)
        return out
   

class VGG16(Model):
    """
    VGG16 
    """
    def __init__(self, num_classes: int=10):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        
        self.block1 = VGGBlock(64, 3, 2, block=1)
        self.block2 = VGGBlock(128, 3, 2, block=2)
        self.block3 = VGGBlock(256, 3, 3, block=3)
        self.block4 = VGGBlock(512, 3, 3, block=4)
        self.block5 = VGGBlock(512, 3, 3, block=5)

        self.flatten = Flatten()
        self.fc1 = Dense(4096, activation="relu")
        self.fc2 = Dense(4096, activation="relu")
        
        self.dropout = Dropout(0.5)
        self.classifier = Dense(self.num_classes, activation="softmax")


    def call(self, inputs: tf.Tensor):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        out = self.classifier(x)
        return out

