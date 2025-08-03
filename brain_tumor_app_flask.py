from flask import Flask, request, jsonify
import numpy as np
import json
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fpdf import FPDF

app = Flask(__name__)

# === Load models and tokenizer ===
classifier_model = load_model("brain_tumor_classifier.h5", compile=False)
segment_model = load_model("tumor_segmentation_model.h5", compile=False)
tokenizer = T5Tokenizer.from_pretrained("vivek-2210/brain-tumor-detector")
t5_model = T5ForConditionalGeneration.from_pretrained("vivek-2210/brain-tumor-detector")

with open("class_indices.json") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

PIXEL_SPACING_CM = 0.0625

# === Utilities ===

def get_gradcam_heatmap(model, img_array, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.output, model.get_layer("Conv_1_bn").output])
    with tf.GradientTape() as tape:
        preds, conv_outputs = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_tumor_info(mask, prob_map):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        area_px = cv2.contourArea(cnt)
        area_cm2 = area_px * (PIXEL_SPACING_CM ** 2)
        confidence = np.mean(prob_map[mask == 255])
        return int(x), int(y), int(w), int(h), round(area_cm2, 2), round(float(confidence) * 100, 2)
    else:
        return None, None, None, None, 0.0, 0.0

def predict_brain_region(x, y, w, h, image_width, image_height):
    center_x = x + w // 2
    center_y = y + h // 2
    hemisphere = "Left Hemisphere" if center_x < image_width / 2 else "Right Hemisphere"
    if center_y < image_height / 3:
        region = "Frontal Lobe"
    elif center_y < 2 * image_height / 3:
        region = "Parietal Region"
    else:
        region = "Occipital/Posterior Region"
    return f"{hemisphere}, {region}"

def generate_hybrid_report(data):
    patient = data['patient_details']
    cls = data['classification']
    seg = data['segmentation']

    t5_input = (
        f"generate report: {patient['name']} aged {patient['age']} has a {cls['predicted_class']} "
        f"located in the **{seg['location_name']}** of the brain with confidence {cls['confidence']}. "
        f"Tumor area is {seg['area_cm2']} cm2."
    )
    input_ids = tokenizer.encode(t5_input, return_tensors="pt")
    output = t5_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    t5_report = tokenizer.decode(output[0], skip_special_tokens=True)

    return t5_report

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    name = request.form.get('name', 'Unknown')
    pid = request.form.get('pid', '0000')
    age = request.form.get('age', '0')

    img = Image.open(file).convert("RGB")
    original_width, original_height = img.size

    # --- Classification ---
    img_resized = img.resize((512, 512))
    img_array = img_to_array(img_resized) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    preds = classifier_model.predict(img_array_exp)
    pred_index = np.argmax(preds)
    pred_class = index_to_class[pred_index]
    confidence = float(preds[0][pred_index])

    # --- Segmentation ---
    seg_img = img.resize((256, 256)).convert('L')
    seg_array = np.expand_dims(img_to_array(seg_img) / 255.0, axis=0)
    prob_map = segment_model.predict(seg_array)[0]
    seg_mask = (prob_map > 0.5).astype(np.uint8) * 255

    x, y, w, h, area_cm2, conf_percent = get_tumor_info(seg_mask, prob_map)

    if None not in (x, y, w, h):
        location_name = predict_brain_region(x, y, w, h, original_width, original_height)
        bbox_info = {"x": x, "y": y, "width": w, "height": h}
    else:
        location_name = "Not Applicable"
        bbox_info = None

    result_json = {
        "filename": file.filename,
        "patient_details": {
            "name": name,
            "id": pid,
            "age": age
        },
        "classification": {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4)
        },
        "segmentation": {
            "segmentation_mask_shape": seg_mask.shape,
            "tumor_area_pixels": int(np.sum(seg_mask > 0)),
            "bounding_box": bbox_info,
            "area_cm2": area_cm2,
            "confidence_percent": conf_percent,
            "location_name": location_name
        }
    }

    report = generate_hybrid_report(result_json)

    return jsonify({
        "prediction": pred_class,
        "confidence": confidence,
        "location": location_name,
        "tumor_area_cm2": area_cm2,
        "t5_report": report
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
