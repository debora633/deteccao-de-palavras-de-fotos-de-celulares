import cv2
import numpy as np
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

summary = {
    "Adaptive Threshold": {
        "imagens": 0,
        "acertos": 0,
        "erros": 0,
        "total_boxes": 0
    },
    "Otsu": {
        "imagens": 0,
        "acertos": 0,
        "erros": 0,
        "total_boxes": 0
    },
    "Global": {
        "imagens": 0,
        "acertos": 0,
        "erros": 0,
        "total_boxes": 0
    }
}


def count_components(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image,
        connectivity=8
    )

    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 20 and area < 2500:
            count += 1

    return count


def save_summary_file(summary_data, path="RESULTADOS.md"):
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Resultados do Projeto\n\n")
        file.write(
            "| Método | Imagens processadas | Imagens com detecção | Imagens sem detecção | Total de boxes | Média de boxes por imagem |\n"
        )
        file.write(
            "|---|---|---|---|---|---|\n"
        )

        for method, values in summary_data.items():
            imagens = values["imagens"]
            acertos = values["acertos"]
            erros = values["erros"]
            total_boxes = values["total_boxes"]
            media_boxes = round(total_boxes / imagens, 2) if imagens else 0
            file.write(
                f"| {method} | {imagens} | {acertos} | {erros} | {total_boxes} | {media_boxes} |\n"
            )


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

    cv2.imwrite(
        os.path.join(ROTACAO_FOLDER, f"{name}_rotated{ext}"),
        image
    )

    image_boxes = image.copy()

    # =========================
    # GRAY
    # =========================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # =========================
    # FILTROS DE RUÍDO
    # =========================

    # 1. GAUSSIAN (SEU ATUAL - principal)
    blur_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. MEDIAN (novo)
    blur_median = cv2.medianBlur(gray, 3)

    # 3. BILATERAL (novo)
    blur_bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # =========================
    # CLAHE (usa Gaussian como base principal)
    # =========================
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    contrast = clahe.apply(blur_gaussian)

    # =========================
    # SHARPEN
    # =========================
    kernel_sharp = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(contrast, -1, kernel_sharp)

    # =========================
    # ILUMINAÇÃO
    # =========================
    background = cv2.GaussianBlur(gray, (101, 101), 0)
    norm = cv2.divide(gray, background, scale=255)

    # =========================
    # BINARIZAÇÃO PRINCIPAL (ADAPTATIVE - INVERTIDA)
    # =========================
    binary = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )
    binary = cv2.bitwise_not(binary)

    # =========================
    # OTSU (NÃO INVERTIDO)
    # =========================
    _, binary_otsu = cv2.threshold(
        norm,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # =========================
    # GLOBAL (NÃO INVERTIDO)
    # =========================
    _, binary_global = cv2.threshold(
        norm,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # =========================
    # CONNECTED COMPONENTS (principal)
    # =========================
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

        if area > 20 and area < 2500:

            cv2.rectangle(
                output_boxes,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                image_boxes,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # =========================
    # CONTAGEM DE COMPONENTES POR MÉTODO
    # =========================
    count_adaptive = count_components(binary)
    count_otsu = count_components(binary_otsu)
    count_global = count_components(binary_global)

    summary["Adaptive Threshold"]["imagens"] += 1
    summary["Adaptive Threshold"]["total_boxes"] += count_adaptive
    if count_adaptive > 0:
        summary["Adaptive Threshold"]["acertos"] += 1
    else:
        summary["Adaptive Threshold"]["erros"] += 1

    summary["Otsu"]["imagens"] += 1
    summary["Otsu"]["total_boxes"] += count_otsu
    if count_otsu > 0:
        summary["Otsu"]["acertos"] += 1
    else:
        summary["Otsu"]["erros"] += 1

    summary["Global"]["imagens"] += 1
    summary["Global"]["total_boxes"] += count_global
    if count_global > 0:
        summary["Global"]["acertos"] += 1
    else:
        summary["Global"]["erros"] += 1

    # =========================
    # SALVAR ETAPAS
    # =========================
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_00_deskew{ext}"), image)

    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_01_gray{ext}"), gray)

    # filtros
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_02_gaussian_blur{ext}"), blur_gaussian)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_03_median_blur{ext}"), blur_median)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_04_bilateral_blur{ext}"), blur_bilateral)

    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_05_clahe{ext}"), contrast)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_06_sharp{ext}"), sharp)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_07_background{ext}"), background)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_08_norm{ext}"), norm)

    # binarizações
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_09_binary_adaptive_inv{ext}"), binary)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_10_binary_otsu{ext}"), binary_otsu)
    cv2.imwrite(os.path.join(ETAPAS_FOLDER, f"{name}_11_binary_global{ext}"), binary_global)

    # =========================
    # RESULTADO FINAL
    # =========================
    cv2.imwrite(
        os.path.join(PROCESSADAS_FOLDER, f"{name}_final{ext}"),
        binary
    )

    # =========================
    # DETECÇÃO
    # =========================
    cv2.imwrite(
        os.path.join(DETECCAO_FOLDER, f"{name}_boxes_binary{ext}"),
        output_boxes
    )

    cv2.imwrite(
        os.path.join(DETECCAO_FOLDER, f"{name}_boxes_original{ext}"),
        image_boxes
    )

    print(f"Salvo: {name}")

save_summary_file(summary)
print("Processamento concluído.")