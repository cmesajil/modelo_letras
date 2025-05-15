import os
import tempfile
import base64
import glob
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, send_file, render_template_string
from skimage import io
from skimage.transform import resize
from tensorflow.keras.models import load_model

# --------------------
# CONFIGURACIÓN FLASK
# --------------------
app = Flask(__name__)
equivalencias = {'ا': 0, 'ب': 1, 'ت': 2}
reverse_equivalencias = {v: k for k, v in equivalencias.items()}
size = (28, 28)

# --------------------
# CARGA DE MODELO
# --------------------
model = load_model("modelo_cnn.h5")  # Asegúrate de tener este archivo en la carpeta

# --------------------
# HTML de dibujo
# --------------------
main_html = """<html> ... tu HTML completo aquí ... </html>"""  # Recorta aquí por simplicidad

# --------------------
# RUTAS FLASK
# --------------------
@app.route("/")
def main():
    return render_template_string(main_html)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
        aleatorio = request.form.get('numero')
        with tempfile.NamedTemporaryFile(delete=False, mode="w+b", suffix='.png', dir=str(aleatorio)) as fh:
            fh.write(base64.b64decode(img_data))
        print("Imagen subida correctamente")
    except Exception as err:
        print("Error al subir imagen:", err)

    return redirect("/", code=302)

@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    images = []
    digits = []
    d = ['ا', 'ب', 'ت']  # Usando las equivalencias en árabe
    for digit in d:
        filelist = glob.glob(f'{digit}/*.png')
        if filelist:
            imgs = io.concatenate_images(io.imread_collection(filelist))
            imgs = imgs[:, :, :, 3]  # Canal alfa
            labels = np.array([digit] * imgs.shape[0])
            images.append(imgs)
            digits.append(labels)
    X = np.vstack(images)
    y = np.concatenate(digits)
    np.save('X.npy', X)
    np.save('y.npy', y)
    return "OK!"

@app.route('/test', methods=['GET'])
def test_model():
    # Cargar datos
    X_raw = np.load('X.npy')
    y = np.load('y.npy')

    # Preprocesamiento
    X_raw = X_raw / 255.
    X = [resize(x, size) for x in X_raw]
    X = np.array(X)
    X = X[..., np.newaxis]  # Añadir canal para CNN

    # Separación en test (aquí simplificamos sin split)
    idx = np.random.choice(X.shape[0], 1)[0]
    im = X[idx]
    label = y[idx]

    # Predicción
    salida = model.predict(im[None, :, :, :])[0]
    pred = salida.argmax()

    # Visualización
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title(f'(test) id:{idx} val:{label}')
    plt.axis('off')
    plt.imshow(-im[:, :, 0], cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title(f"Predicción: {pred} ({reverse_equivalencias[pred]}) vs {label}")
    plt.ylabel("Probabilidad")
    plt.xlabel("Clase")
    plt.ylim([0, 1])
    plt.bar(np.arange(3), salida, tick_label=list(equivalencias.keys()))
    plt.tight_layout()

    # Guardar resultado en imagen temporal
    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_img.name)
    plt.close()

    return send_file(temp_img.name, mimetype='image/png')

@app.route('/X.npy')
def download_X():
    return send_file('./X.npy')

@app.route('/y.npy')
def download_y():
    return send_file('./y.npy')

# --------------------
# INIT APP
# --------------------
if __name__ == "__main__":
    for d in equivalencias.keys():
        if not os.path.exists(d):
            os.mkdir(d)
    app.run(debug=True)
