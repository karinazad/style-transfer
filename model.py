import tensorflow as tf
from tensorflow.keras.applications import vgg19

from losses import *
from utils.image import preprocess_image, deprocess_image


class StyleTransfer():
    def __init__(self, base_image_path = BASE_IMAGE_PATH , style_image_path = STYLE_IMAGE_PATH,
                 optimizer = None, base_model = None):

        assert type(base_image_path) == str and type(style_image_path) == str

        if optimizer is None:
            self.optimizer = keras.optimizers.SGD(
                keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))
        else:
            self.optimizer = optimizer

        if base_model is None:
            self.model = vgg19.VGG19(weights="imagenet", include_top=False)
        else:
            self.model = base_model

        self.base_image = preprocess_image(base_image_path, self.model)
        self.style_image = preprocess_image(style_image_path, self.model)
        self.combined_image = tf.Variable(preprocess_image(base_image_path, self.model))


        self.style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1", ]

        self.content_layer_name = "block5_conv2"
        self.outputs_dict = dict([(layer.name, layer.output) for layer in self.model.layers])
        self.feature_extractor = keras.Model(inputs=self.model.inputs, outputs=self.outputs_dict)


    @tf.function
    def compute_loss_and_grads(self):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(self.combined_image, self.base_image, self.style_image)
        grads = tape.gradient(loss, self.combined_image)
        return loss, grads

    def compute_loss(self,):
        input_tensor = tf.concat(
            [self.base_image, self.style_image, self.combined_image], axis=0
        )
        features = self.feature_extractor(input_tensor)

        # Initialize the loss
        loss = tf.zeros(shape=())

        # Add content loss
        layer_features = features[self.content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + content_weight * content_loss(
            base_image_features, combination_features)
        # Add style loss
        for layer_name in self.style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (style_weight / len(self.style_layer_names)) * sl

        # Add total variation loss
        loss += total_variation_weight * total_variation_loss(self.combined_image)
        return loss


    def train(self, epochs = 10000):
        for i in range(epochs):
            loss, grads = self.compute_loss_and_grads(self.combined_image, self.base_image, self.style_image)

            self.optimizer.apply_gradients([(grads, self.combined_image)])

            if i % 100 == 0:
                print("Iteration %d: loss=%.2f" % (i, loss))
                img = deprocess_image(self.combined_image.numpy())
                fname = result_prefix + "_at_iteration_%d.png" % i
                keras.preprocessing.image.save_img(fname, img)



