import streamlit as st
from PIL import Image
import numpy as np
import random
import torch
from torchsummary import summary
import torchvision.transforms as transforms
from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, non_max_suppression, rescale_boxes
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def process_image(img, model, classes, img_size=416, conf_thres=0.5, nms_thres=0.5):
    model.eval()  # Set model to evaluation mode
    input_img = transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])(
        (img, np.zeros((1, 5)))
    )[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, img.shape[:2])

    return detections


def draw_detections(image, detections, classes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
        bbox_colors = random.sample(colors, n_cls_preds)

        for x1, y1, x2, y2, conf, cls_pred in detections:
            box_w = x2 - x1
            box_h = y2 - y1

            # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            if int(cls_pred) == 0:
                color = "purple"
            bbox = patches.Rectangle(
                (x1, y1),
                box_w,
                box_h,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(bbox)
            plt.text(
                x1,
                y1,
                s=f"{classes[int(cls_pred)]}: {conf:.2f}",
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )
    plt.axis("off")
    return fig


model_path = "config/yolov3-custom.cfg"
weights_path = "checkpoints/yolov3_ckpt_65.pth"
classes_path = "data/classes.names"
model = load_model(model_path, weights_path)
print(model)
classes = load_classes(classes_path)

st.title("Object Detection with YOLOv3")

conf_thres = st.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
)
nms_thres = st.slider(
    "NMS Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    detections = process_image(
        img_array, model, classes, conf_thres=conf_thres, nms_thres=nms_thres
    )

    fig = draw_detections(img_array, detections, classes)

    st.pyplot(fig)
