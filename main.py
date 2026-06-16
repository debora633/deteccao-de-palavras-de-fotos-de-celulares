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
DETECCAO_FOLDER = "dataset/deteccao"
ROTACAO_FOLDER = "dataset/originais_rotacao"

os.makedirs(PROCESSADAS_FOLDER, exist_ok=True)
os.makedirs(ETAPAS_FOLDER, exist_ok=True)
os.makedirs(DETECCAO_FOLDER, exist_ok=True)
os.makedirs(ROTACAO_FOLDER, exist_ok=True)

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

    print(f"Processando: {filename}")

    name, ext = os.path.splitext(filename)

    # DESKEW
    angle = get_angle(image)
    image = rotate(image, angle)

    # salvar rotação
    cv2.imwrite(
        os.path.join(ROTACAO_FOLDER, f"{name}_rotated{ext}"),
        image
    )

    # cópia ORIGINAL para desenhar caixas depois
    image_boxes = image.copy()

    # PREPROCESSAMENTO
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    contrast = clahe.apply(blur)

    kernel_sharp = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(contrast, -1, kernel_sharp)


    # CORREÇÃO DE ILUMINAÇÃO
    background = cv2.GaussianBlur(gray, (101, 101), 0)
    norm = cv2.divide(gray, background, scale=255)

    # BINARIZAÇÃO
    binary = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    binary = cv2.bitwise_not(binary)

    # CONNECTED COMPONENTS
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    output_boxes = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # filtro melhorado
        if area > 20 and area < 2500:

            # caixa na imagem binária
            cv2.rectangle(
                output_boxes,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # caixa na imagem ORIGINAL 
            cv2.rectangle(
                image_boxes,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
    # SALVAR ETAPAS
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_00_deskew{ext}"), image)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_01_gray{ext}"), gray)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_02_blur{ext}"), blur)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_03_clahe{ext}"), contrast)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_04_sharp{ext}"), sharp)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_05_background{ext}"), background)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_06_norm{ext}"), norm)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_07_binary{ext}"), binary)

    # SALVAR RESULTADO FINAL
    cv2.imwrite(
        os.path.join(PROCESSADAS_FOLDER, f"{name}_final{ext}"),
        binary
    )

    # SALVAR DETECÇÃO
    cv2.imwrite(
        os.path.join(DETECCAO_FOLDER, f"{name}_boxes_binary{ext}"),
        output_boxes
    )


    # SALVAR DETECÇÃO 
    cv2.imwrite(
        os.path.join(DETECCAO_FOLDER, f"{name}_boxes_original{ext}"),
        image_boxes
    )

    print(f"Salvo: {name}")

print("Processamento concluído.")