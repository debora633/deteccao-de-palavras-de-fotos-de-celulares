import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from jdeskew.estimator import get_angle
from jdeskew.utility import rotate

# PASTAS
INPUT_FOLDER = "dataset/originais"
OUTPUT_FOLDER = "dataset/processadas"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def show(title, image, cmap='gray'):

    plt.figure(figsize=(10, 7))
    plt.title(title)

    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')
    plt.show()

valid_extensions = (".jpg", ".jpeg", ".png")

images = [
    file for file in os.listdir(INPUT_FOLDER)
    if file.lower().endswith(valid_extensions)
]

for filename in images:

    image_path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erro ao carregar: {filename}")
        continue

    # UPSCALE
    scale = 3
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    print(f"Processando: {filename}")

    # =========================
    # DESKEW (ROTAÇÃO AUTOMÁTICA)
    # =========================
    angle = get_angle(image)
    image = rotate(image, angle)

    # ESCALA DE CINZA
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(blur)

    # SHARPEN
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharp = cv2.filter2D(contrast, -1, kernel)

    # ILUMINAÇÃO
    background = cv2.GaussianBlur(sharp, (101, 101), 0)
    norm = cv2.divide(sharp, background, scale=255)

    # BINARIZAÇÃO
    binary = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    # MORFOLOGIA
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel, iterations=3)
    dilated = cv2.dilate(closed, morph_kernel, iterations=1)

    # CONTORNOS
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w > 20 and h > 10:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # SALVAR
    name, ext = os.path.splitext(filename)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_upscale{ext}"), image)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_gray{ext}"), gray)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_clahe{ext}"), contrast)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_sharp{ext}"), sharp)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_morph{ext}"), dilated)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_binary{ext}"), binary)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_final{ext}"), result)

    # MOSTRAR
    show("Original (Deskew aplicado)", image)
    show("Imagem Binaria", binary)

print("Processamento concluído.")