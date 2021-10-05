import tensorflow as tf
import tensorflow_datasets as tfds
from model.transformer import ViT

def main():
    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()
    mnist_train = mnist_builder.as_dataset(split='train')
    # print(mnist_train)
    vit = ViT()

if __name__ == '__main__':
    main()