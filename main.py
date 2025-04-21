import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import io, color
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = io.imread(img_path)
            if img is not None:
                images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images

def extract_features(image):
    # Resize image to a fixed size
    image = resize(image, (128, 128))  # Resize to 128x128 pixels
    # Convert RGBA to RGB if the image has an alpha channel
    if image.shape[2] == 4:
        image = image[:, :, :3]
    # Convert to grayscale if the image is multichannel
    if len(image.shape) == 3:
        image = rgb2gray(image)
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              visualize=True)
    return features

def prepare_data(image_folder):
    images = load_images_from_folder(image_folder)
    features = [extract_features(img) for img in images]
    labels = [i for i in range(len(images))]  # Simplified labeling for demonstration
    return np.array(features), np.array(labels)

def train_model(features, labels):
    model = make_pipeline(StandardScaler(), SVC(probability=True))
    model.fit(features, labels)
    return model

def match_fingerprint(model, test_image):
    test_features = extract_features(test_image).reshape(1, -1)
    probabilities = model.predict_proba(test_features)
    return probabilities

def visualize_matching(image, probabilities, folder):
    match_percentage = np.max(probabilities) * 100
    best_match_idx = np.argmax(probabilities)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Test Fingerprint')
    
    best_match_image = io.imread(os.path.join(folder, os.listdir(folder)[best_match_idx]))
    ax[1].imshow(best_match_image, cmap='gray')
    ax[1].set_title(f'Best Match: {match_percentage:.2f}%')
    
    plt.show()

def main(training_folder, test_image_path):
    features, labels = prepare_data(training_folder)
    model = train_model(features, labels)
    
    test_image = io.imread(test_image_path)
    probabilities = match_fingerprint(model, test_image)
    
    visualize_matching(test_image, probabilities, training_folder)

# Example usage
training_folder = r'D:\cg\train'
test_image_path = r'D:\cg\test\1_image1.jpg'
main(training_folder, test_image_path)
