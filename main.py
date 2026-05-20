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
    # REDUÇÃO DE RUÍDO
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
        [-1, 5,-1],
        [0, -1, 0]
    ])

    sharp = cv2.filter2D(contrast, -1, kernel)

    # GERAR NOME
    name, ext = os.path.splitext(filename)

    output_name = f"{name}_processada{ext}"

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )
    # SALVAR
    cv2.imwrite(output_path, sharp)

    print(f"Salva em: {output_path}")

print("Processamento concluído.")