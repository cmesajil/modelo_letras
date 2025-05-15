import os
import base64
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file
from skimage import io
from skimage.transform import resize
from tensorflow.keras.models import load_model

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
@app.route("/", methods=["GET", "POST"])
def main():
    pred_letra = None
    pred_img = None

    if request.method == "POST":
        try:
            img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=UPLOAD_DIR) as fh:
                fh.write(base64.b64decode(img_data))
                img_path = fh.name

            # Leer y procesar imagen
            img = io.imread(img_path)[..., 3] / 255.0  # canal alfa normalizado
            img = resize(img, size)
            img = img[np.newaxis, ..., np.newaxis]

            salida = model.predict(img)[0]
            pred = salida.argmax()
            pred_letra = reverse_equivalencias[pred]

            # Visualización
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(-img[0, :, :, 0], cmap="gray")
            plt.title("Imagen Dibujada")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.bar(np.arange(len(equivalencias)), salida, tick_label=list(equivalencias.keys()))
            plt.title(f"Predicción: {pred_letra}")
            plt.ylim([0, 1])
            plt.tight_layout()

            temp_plot = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.savefig(temp_plot.name)
            plt.close()

            pred_img = os.path.basename(temp_plot.name)
        except Exception as e:
            print("Error procesando imagen:", e)

    return render_template("index.html", pred=pred_letra, img=pred_img)

@app.route("/pred_img/<filename>")
def pred_img(filename):
    return send_file(os.path.join(tempfile.gettempdir(), filename), mimetype="image/png")

# --------------------
# INIT APP
# --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
