import os
import cv2
import matplotlib.pyplot as plt

def visualize_samples(dataset_path='dice_dataset', num_samples=6):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for number in range(1, num_samples + 1):
        class_dir = os.path.join(dataset_path, str(number))
        images = os.listdir(class_dir)
        if images:
            img_path = os.path.join(class_dir, images[0])
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[number-1].imshow(img_rgb)
            axes[number-1].set_title(f'Число: {number}')
            axes[number-1].axis('off')

    plt.tight_layout()
    plt.show()

visualize_samples()