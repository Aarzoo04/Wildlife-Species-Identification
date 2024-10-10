from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
from ultralytics import YOLO
import cv2

app = FastAPI()

UPLOAD_DIRECTORY = "api\uploaded_images"
PROCESSED_IMAGES_DIRECTORY = "api\processed_images"

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_DIRECTORY, exist_ok=True)

onnx_model = YOLO("best.onnx", task='detect')

class_names = ["Striped_Hyena", "Fishing_Cat"]

def plot_bbox_and_save(image_path, bbox_coords, class_name):
    img = cv2.imread(image_path)
    for bbox in bbox_coords:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    class_folder = os.path.join(PROCESSED_IMAGES_DIRECTORY, class_name)
    os.makedirs(class_folder, exist_ok=True)

    base_name = os.path.basename(image_path)
    output_path = os.path.join(class_folder, base_name)
    cv2.imwrite(output_path, img)
    return output_path

@app.post("/upload/")
async def upload_images(files: list[UploadFile] = File(...)):
    try:
        processed_images = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            results = onnx_model(file_path)

            any_detections = False
            contains_fishing_cat = False
            contains_striped_hyena = False

            for result in results:
                detections_by_class = {name: [] for name in class_names}
                for i, box in enumerate(result.boxes):
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    xyxy = box.xyxy.cpu().numpy().tolist()[0]

                    if conf > 0.5:
                        class_name = class_names[cls] if cls < len(class_names) else "Unknown"
                        detections_by_class[class_name].append(xyxy)
                        any_detections = True

                        # Check for class detections
                        if class_name == "Fishing_Cat":
                            contains_fishing_cat = True
                        elif class_name == "Striped_Hyena":
                            contains_striped_hyena = True

                # Determine the folder to save the image based on detections
                if any_detections:
                    if contains_striped_hyena:
                        # Save to Striped Hyena folder (no need to save in Fishing Cat if both are detected)
                        output_path = plot_bbox_and_save(file_path, detections_by_class["Striped_Hyena"], "Striped_Hyena")
                        processed_images.append(output_path)
                    elif contains_fishing_cat:
                        # Save only in the Fishing Cat folder if no Striped Hyena is detected
                        output_path = plot_bbox_and_save(file_path, detections_by_class["Fishing_Cat"], "Fishing_Cat")
                        processed_images.append(output_path)
                else:
                    # No detections, save in 'other' folder
                    other_folder = os.path.join(PROCESSED_IMAGES_DIRECTORY, "other")
                    os.makedirs(other_folder, exist_ok=True)
                    base_name = os.path.basename(file_path)
                    output_path = os.path.join(other_folder, base_name)
                    shutil.copy(file_path, output_path)
                    processed_images.append(output_path)

        return JSONResponse(content={"message": "Files processed successfully", "processed_images": processed_images})
    except Exception as e:
        return JSONResponse(content={"message": f"Error uploading files: {str(e)}"}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "ONNX Model API is up and running"}
