import os
import tempfile
import base64
import glob
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, send_file, render_template
from skimage import io
from skimage.transform import resize
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# --------------------
# CONFIGURACIÓN FLASK
# --------------------
app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

equivalencias = {'ا': 0, 'ب': 1, 'ت': 2}
reverse_equivalencias = {v: k for k, v in equivalencias.items()}
size = (28, 28)

# --------------------
# CARGA DE MODELO
# --------------------
model = load_model("modelo_cnn.h5")

# --------------------
# RUTAS FLASK
# --------------------
@app.route("/")
def main():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    try:
        img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
        label = secure_filename(request.form.get('numero', '0'))

        label_dir = os.path.join(UPLOAD_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, mode="w+b", suffix='.png', dir=label_dir) as fh:
            fh.write(base64.b64decode(img_data))

        print("Imagen subida correctamente")
    except Exception as err:
        print("Error al subir imagen:", err)

    return redirect("/", code=302)

@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    images = []
    digits = []
    for digit in equivalencias.keys():
        path = os.path.join(UPLOAD_DIR, digit)
        filelist = glob.glob(os.path.join(path, '*.png'))
        if not filelist:
            continue
        imgs = io.concatenate_images(io.imread_collection(filelist))
        imgs = imgs[:, :, :, 3]  # Canal alfa
        labels = np.array([digit] * imgs.shape[0])
        images.append(imgs)
        digits.append(labels)

    if not images:
        return "No se encontraron imágenes para preparar."

    X = np.vstack(images)
    y = np.concatenate(digits)
    np.save('data/X.npy', X)
    np.save('data/y.npy', y)
    return "OK!"

@app.route('/test', methods=['GET'])
def test_model():
    try:
        X_raw = np.load('data/X.npy')
        y = np.load('data/y.npy')
    except Exception as e:
        return f"Error cargando datos: {e}"

    X_raw = X_raw / 255.
    X = np.array([resize(x, size) for x in X_raw])
    X = X[..., np.newaxis]

    idx = np.random.choice(X.shape[0])
    im = X[idx]
    label = y[idx]

    salida = model.predict(im[None, :, :, :])[0]
    pred = salida.argmax()

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
    plt.bar(np.arange(len(equivalencias)), salida, tick_label=list(equivalencias.keys()))
    plt.tight_layout()

    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_img.name)
    plt.close()

    return send_file(temp_img.name, mimetype='image/png')

@app.route('/X.npy')
def download_X():
    return send_file('data/X.npy')

@app.route('/y.npy')
def download_y():
    return send_file('data/y.npy')

# --------------------
# INIT APP
# --------------------
if __name__ == "__main__":
    for letra in equivalencias.keys():
        os.makedirs(os.path.join(UPLOAD_DIR, letra), exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
