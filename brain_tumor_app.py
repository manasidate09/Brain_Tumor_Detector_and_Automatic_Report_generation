import streamlit as st
import numpy as np
import json
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fpdf import FPDF
import base64
import textwrap

# === Load Models ===
classifier_model = load_model("brain_tumor_classifier.keras", compile=False)
segment_model = load_model("tumor_segmentation_model.h5", compile=False)
tokenizer = T5Tokenizer.from_pretrained("manasivivek/tumordetection")
t5_model = T5ForConditionalGeneration.from_pretrained("manasivivek/tumordetection")

# === Load class indices ===
with open("class_indices.json") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

PIXEL_SPACING_CM = 0.0625

# Grad-CAM utility
def get_gradcam_heatmap(model, img_array, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-1).output, model.get_layer("Conv_1_bn").output])
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
        confidence_percent = round(float(confidence) * 100, 2)
        return int(x), int(y), int(w), int(h), round(area_cm2, 2), confidence_percent
    else:
        return None, None, None, None, 0.0, 0.0

def generate_hybrid_report(json_data):
    patient = json_data['patient_details']
    cls = json_data['classification']
    seg = json_data['segmentation']

    # T5 Input Construction
    t5_input_old = f"generate report: {patient['name']} aged {patient['age']} has a {cls['predicted_class']} with confidence {cls['confidence']}. Tumor area is {seg['area_cm2']} cm2."
    t5_input = f"generate report: {patient['name']} aged {patient['age']} has a {cls['predicted_class']}tumor located in the **{seg['location_name']}** of the brain with confidence {cls['confidence']}. Tumor area is {seg['area_cm2']} cm2."
    input_ids = tokenizer.encode(t5_input, return_tensors="pt")
    output = output = t5_model.generate(
    input_ids,
    max_length=512,
    num_beams=4,
    no_repeat_ngram_size=3,
    repetition_penalty=2.5,
    length_penalty=1.0,
    early_stopping=True
    )
    t5_report = tokenizer.decode(output[0], skip_special_tokens=True)

    # Custom Interpretation & Recommendation
    def get_custom_insights(tumor_class, seg_info):
        tumor_class = tumor_class.lower()
        area = seg_info.get("area_cm2", 0.0)
        bbox = seg_info.get("bounding_box", None)
        
        interpretations = {
            "glioma_tumor" : (
                f"The MRI scan reveals the presence of a **Glioma tumor**,\n\n a type of primary brain neoplasm arising from glial cells. "
                f"\n\n These tumors may exhibit infiltrative and aggressive growth patterns. "
                f"\n\n In this case, the lesion occupies approximately **{area} cm²**, "
                f"\n\n indicating a possible mass effect on surrounding brain tissue."
            ),
            "meningioma_tumor" : (
                f"\n\n A **Meningioma tumor** has been detected, likely originating from the meninges. "
                f"\n\n Though often benign, its size (~**{area} cm²**) and location should be "
                f"\n\n evaluated for compressive effects on neural structures."
            ),
            "pituitary_tumor" : (
                f"The scan suggests the presence of a **Pituitary adenoma**, typically"
                 f" arising from hormone-secreting cells. "
                f"With a size of roughly **{area} cm²**, the lesion may affect hormonal balance" 
                f"and vision depending on its growth pattern."
            ),
            "no_tumor" : (
                "No abnormal tumor structures were detected in this MRI scan. Brain parenchyma appears normal, \n\n with no masses or lesions observed."
            )
        }   
    

        recommendations = {
            "glioma_tumor": (
                "Recommend prompt **neuro-oncology referral**. Consider **MRI with contrast**, **biopsy**, \n\n and **histopathological grading**. "
                "Treatment options may include **surgical resection**, **radiotherapy**, \n\n and/or **chemotherapy** depending on tumor type and grade."
            ),
            "meningioma_tumor": (
                "Suggest **periodic imaging follow-up**. If symptomatic or showing growth, consult \n\n **neurosurgery** for potential resection. "
                "Stereotactic radiosurgery may  be considered for select cases."
            ),
            "pituitary_tumor": (
                "Recommend **endocrinological workup**, including hormonal panels and **visual field testing**. "
                "Referral to a **neurosurgeon** is advised for further evaluation and management planning."
            ),
            "no_tumor": (
                "No further medical intervention is required based on this scan. Recommend routine \n\n health monitoring and follow-up if symptoms develop."
            )
        }

        disclaimer = (
            "**Disclaimer:** 🧠 This is an AI-generated medical report based on uploaded MRI scans. "
            "\n\n It is not a substitute for professional medical advice, diagnosis, or treatment. "
            "\n\n Always consult a certified medical specialist for clinical decisions."
        )

        interpretation = interpretations.get(tumor_class, "Interpretation not available for the detected class.")
        recommendation = recommendations.get(tumor_class, "General neurological consultation is advised.")

        return interpretation, recommendation, disclaimer



    interpretation, recommendation, disclaimer = get_custom_insights(cls['predicted_class'], seg)
    itn=textwrap.fill(interpretation, width=100)
    full_report = (
        f"Patient Name: {patient['name']} \n\n"
        f"Age: {patient['age']} \n\n"
        f"Patient ID: {patient['id']} \n\n"
        f"--- \n\n"
        f"**T5 Report: **\n\n {t5_report} \n\n"
        f"--- \n\n"
        f"**Interpretation: **\n\n {itn} \n\n"
       # f"**Recommendation: **\n\n{recommendation}\n\n"
        f"**Recommendation: ** \n\n {recommendation} \n\n"
        f"{disclaimer}"
    )
    return full_report

def save_report_to_pdf(report_text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    report_text = report_text.encode('latin-1', 'ignore').decode('latin-1')
    wrapped_text = textwrap.fill(report_text, width=100)  # Adjust width as needed
    pdf.multi_cell(0, 10, wrapped_text)
    #for line in report_text.split('\n'):
        #pdf.multi_cell(0, 10, line)
    
    save_path = os.path.join("reports", filename)
    os.makedirs("reports", exist_ok=True)
    pdf.output(save_path)
    return save_path

    # --- Approximate Brain Region ---
def predict_brain_region(x, y, w, h, image_width, image_height):
    center_x = x + w // 2
    center_y = y + h // 2

    # Approximate hemisphere
    hemisphere = "Left Hemisphere" if center_x < image_width / 2 else "Right Hemisphere"

    # Approximate anterior/posterior region
    if center_y < image_height / 3:
        region = "Frontal Lobe"
    elif center_y < 2 * image_height / 3:
        region = "Parietal Region"
    else:
        region = "Occipital/Posterior Region"

    return f"{hemisphere}, {region}"

def display_download_button(filepath):
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(filepath)}"> Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

# === Streamlit UI ===
st.title(" Brain Tumor Detection + Hybrid Report Generator")

with st.form("patient_form"):
    name = st.text_input("Patient Name")
    pid = st.text_input("Patient ID")
    age = st.number_input("Patient Age", min_value=0, max_value=120, step=1)
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    submitted = st.form_submit_button("Analyze & Generate Report")

if submitted and uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    original_width, original_height = img.size

    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((512, 512))
    img_array = img_to_array(img_resized) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    preds = classifier_model.predict(img_array_exp)
    pred_index = np.argmax(preds)
    pred_class = index_to_class[pred_index]
    confidence = float(preds[0][pred_index])

    st.subheader(" Classification Result")
    st.success(f"Prediction: **{pred_class}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    heatmap = get_gradcam_heatmap(classifier_model, img_array_exp, pred_index)
    heatmap_resized = cv2.resize(heatmap, (512, 512))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_color, 0.4, 0)
    st.image(superimposed_img, caption="Grad-CAM Visualization", use_container_width=True)

    seg_img = img.resize((256, 256)).convert('L')
    seg_array = np.expand_dims(img_to_array(seg_img) / 255.0, axis=0)

    prob_map = segment_model.predict(seg_array)[0]
    seg_mask = (prob_map > 0.5).astype(np.uint8) * 255

    seg_mask_resized = cv2.resize(seg_mask, (512, 512))
    seg_mask_rgb = cv2.cvtColor(seg_mask_resized, cv2.COLOR_GRAY2RGB)
    segmented_overlay = cv2.addWeighted(np.array(img_resized), 0.6, seg_mask_rgb, 0.4, 0)
    st.image(segmented_overlay, caption="Segmentation Mask Overlay", use_container_width=True)

    x, y, w, h, area_cm2, conf_percent = get_tumor_info(seg_mask, prob_map)
    location_name = None
    location_name = predict_brain_region(x, y, w, h, original_width, original_height)
    bbox_info = {"x": x, "y": y, "width": w, "height": h} if None not in (x, y, w, h) else None

    result_json = {
        "filename": uploaded_file.name,
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
            "location_name":location_name
        }
    }

    full_report = generate_hybrid_report(result_json)
    st.markdown("###  Hybrid AI Medical Report")
    st.text(full_report)

    pdf_path = save_report_to_pdf(full_report, f"{pid}_{name}_report.pdf")
    display_download_button(pdf_path)

