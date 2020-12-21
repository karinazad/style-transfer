from tensorflow import keras

BASE_IMAGE_PATH = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
STYLE_IMAGE_PATH = keras.utils.get_file(
    "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
)
result_prefix = "paris_generated"

# Weights of the different loss components
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# Dimensions of the generated picture.
width, height = keras.preprocessing.image.load_img(BASE_IMAGE_PATH).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
