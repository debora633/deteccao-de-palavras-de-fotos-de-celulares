import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate

# PASTAS
INPUT_FOLDER = "dataset/originais"
PROCESSADAS_FOLDER = "dataset/processadas"
ETAPAS_FOLDER = "dataset/etapas"

# cria as pastas se não existirem
os.makedirs(PROCESSADAS_FOLDER, exist_ok=True)
os.makedirs(ETAPAS_FOLDER, exist_ok=True)

# FUNÇÃO PARA MOSTRAR IMAGENS
def show(title, image, cmap='gray'):

    plt.figure(figsize=(10, 7))
    plt.title(title)

    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')
    plt.show()

# PEGAR TODAS AS IMAGENS
valid_extensions = (".jpg", ".jpeg", ".png")

images = [
    file for file in os.listdir(INPUT_FOLDER)
    if file.lower().endswith(valid_extensions)
]

# PROCESSAR TODAS AS IMAGENS
for filename in images:

    image_path = os.path.join(INPUT_FOLDER, filename)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Erro ao carregar: {filename}")
        continue
    # =========================
    # DESKEW (ROTAÇÃO AUTOMÁTICA)
    # =========================
    angle = get_angle(image)
    image = rotate(image, angle)

    print(f"Processando: {filename}")

    # ESCALA DE CINZA
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # BLUR
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(blur)

    # SHARPEN
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharp = cv2.filter2D(contrast, -1, kernel)

    # CORREÇÃO DE ILUMINAÇÃO
    background = cv2.GaussianBlur(gray, (101, 101), 0)

    norm = cv2.divide(
        gray,
        background,
        scale=255
    )

    # BINARIZAÇÃO ADAPTATIVA
    binary = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    # =========================
    # SALVAR
    # =========================

    name, ext = os.path.splitext(filename)

    # ETAPAS
    cv2.imwrite(
        os.path.join(ETAPAS_FOLDER, f"{name}_00_deskew{ext}"),
        image
    )
    cv2.imwrite(
        os.path.join(ETAPAS_FOLDER, f"{name}_01_gray{ext}"),
        gray
    )

    cv2.imwrite(
        os.path.join(ETAPAS_FOLDER, f"{name}_02_blur{ext}"),
        blur
    )

    cv2.imwrite(
        os.path.join(ETAPAS_FOLDER, f"{name}_03_clahe{ext}"),
        contrast
    )

    cv2.imwrite(
        os.path.join(ETAPAS_FOLDER, f"{name}_04_sharp{ext}"),
        sharp
    )

    cv2.imwrite(
        os.path.join(ETAPAS_FOLDER, f"{name}_05_background{ext}"),
        background
    )

    cv2.imwrite(
        os.path.join(ETAPAS_FOLDER, f"{name}_06_norm{ext}"),
        norm
    )

    cv2.imwrite(
        os.path.join(ETAPAS_FOLDER, f"{name}_07_binary{ext}"),
        binary
    )

    # RESULTADO FINAL
    output_name = f"{name}_final{ext}"
    output_path = os.path.join(
        PROCESSADAS_FOLDER,
        output_name
    )

    cv2.imwrite(output_path, binary)

    print(f"Salva em: {output_path}")

print("Processamento concluído.")