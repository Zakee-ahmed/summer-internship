import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow_hub as hub

def load_img(path_to_img, max_dim=512):
    img = Image.open(path_to_img)
    img = img.convert('RGB')
    img.thumbnail((max_dim, max_dim))
    img = np.array(img)
    img = img[tf.newaxis, :]
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

def run_style_transfer(content_path, style_path):
    # Load images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # Load pre-trained model from TensorFlow Hub
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Perform style transfer
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    # Display results
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title("Content Image")
    plt.imshow(np.squeeze(content_image))
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Style Image")
    plt.imshow(np.squeeze(style_image))
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Stylized Image")
    plt.imshow(np.squeeze(stylized_image))
    plt.axis('off')
    plt.show()

    # Save output
    out_img = tensor_to_image(stylized_image)
    out_img.save("stylized_output.png")
    print("Stylized image saved as 'stylized_output.png'.")

if _name_ == '_main_':
    content_path = 'your_content.jpg'  # Replace with your image file
    style_path = 'your_style.jpg'      # Replace with your style file
    run_style_transfer(content_path, style_path)
