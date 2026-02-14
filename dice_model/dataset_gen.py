import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt

class DiceDatasetGenerator:
    def __init__(self, image_size=128):
        self.image_size = image_size
        self.dice_config = {
            1: [(0.5, 0.5)],
            2: [(0.25, 0.25), (0.75, 0.75)],
            3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
            4: [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)],
            5: [(0.25, 0.25), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (0.75, 0.75)],
            6: [(0.25, 0.25), (0.25, 0.5), (0.25, 0.75),
                (0.75, 0.25), (0.75, 0.5), (0.75, 0.75)]
        }

    def create_dice_face(self, number):
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

        face_color = (255, 255, 255)
        img[:, :] = face_color

        dot_color = (0, 0, 0)

        dot_radius = 10

        for x_rel, y_rel in self.dice_config[number]:
            x = int(x_rel * self.image_size)
            y = int(y_rel * self.image_size)

            cv2.circle(img, (x, y), dot_radius, dot_color, -1)
            cv2.circle(img, (x-2, y-2), dot_radius//3, (255, 255, 255), -1)

        border_color = (100, 100, 100)
        border_thickness = 1
        cv2.rectangle(img, (2, 2), (self.image_size-3, self.image_size-3),
                     border_color, border_thickness)

        return img

    def generate_dataset(self, samples_per_class=6, output_dir='dice_dataset'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for number in range(1, 7):
            class_dir = os.path.join(output_dir, str(number))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            print(f"Генерация изображений для числа {number}...")
            for i in range(samples_per_class):
                img = self.create_dice_face(number)
                img_path = os.path.join(class_dir, f'dice_{number}_{i:04d}.png')
                cv2.imwrite(img_path, img)

        print(f"Датасет успешно создан в директории {output_dir}")

generator = DiceDatasetGenerator()
generator.generate_dataset(samples_per_class=800, output_dir='dice_dataset')
