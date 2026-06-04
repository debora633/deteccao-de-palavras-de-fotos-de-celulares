import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# PASTAS
INPUT_FOLDER = "dataset/originais"
OUTPUT_FOLDER = "dataset/processadas"

# cria a pasta se ela não existir
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

    # caminho completo da imagem
    image_path = os.path.join(INPUT_FOLDER, filename)

    # leitura
    image = cv2.imread(image_path)

    # verifica se carregou
    if image is None:
        print(f"Erro ao carregar: {filename}")
        continue

    print(f"Processando: {filename}")

    # ESCALA DE CINZA
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # CLAHE - MELHORA CONTRASTE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(blur)

    # SHARPEN - REALCE DE BORDAS
    kernel = np.array([
    [0, -1, 0],
    [-1, 5,-1],
    [0, -1, 0]
    ])

    sharp = cv2.filter2D(contrast, -1, kernel)

    #CORREÇÃO ILUMINAÇÃO
    background = cv2.GaussianBlur(sharp, (101,101), 0)
    norm = cv2.divide(sharp, background, scale=255)
    
    # BINARIZAÇÃO ADAPTATIVA GAUSSIANA
    binary = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    #MORFOLOGIA
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # fecha falhas nas letras
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

    # junta regiões próximas
    dilated = cv2.dilate(closed, morph_kernel, iterations=1)

    #CONTORNOS
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cópia para desenhar caixas
    result = image.copy()

    #BOUNDING BOXES
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        # remove ruídos pequenos
        if w > 20 and h > 10:
            cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2)

    # SALVAR
    name, ext = os.path.splitext(filename)
    # imagem em escala de cinza
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_gray{ext}"),gray)

    # imagem após CLAHE (contraste)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER,f"{name}_clahe{ext}"),contrast)

    # imagem após sharpen (nitidez)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER,f"{name}_sharp{ext}"),sharp)

    # imagem após morfologia
    cv2.imwrite(os.path.join(OUTPUT_FOLDER,f"{name}_morph{ext}"),dilated)

    # imagem binarizada
    cv2.imwrite(os.path.join(OUTPUT_FOLDER,f"{name}_binary{ext}"),binary)

    # imagem final com bounding boxes
    cv2.imwrite(os.path.join(OUTPUT_FOLDER,f"{name}_final{ext}"),result)

    # MOSTRAR RESULTADO
    show("Original", image)
    show("CLAHE - Contraste Melhorado", contrast)
    show("Sharpen - Nitidez", sharp)
    show("Imagem Binaria", binary)
    show("Texto Detectado", result)

print("Processamento concluído.")