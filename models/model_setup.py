from tensorflow.keras.applications import DenseNet201, MobileNetV2, VGG19, ResNet50, InceptionV3
transfer_models={
    "Densenet" : DenseNet201,
    "Mobilenet": MobileNetV2,
    "VGG": VGG19,
    "Resnet": ResNet50,
    "Inception": InceptionV3
}