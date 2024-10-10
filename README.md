# Wildlife Species Identification

## Overview
Wildlife Species Identification is a machine-learning project aimed at identifying and classifying two species of wildlife: the **Striped Hyena and the Fishing Cat**. Utilizing a custom-trained YOLOv8 model, this application achieves an impressive accuracy of **98.72%** in detecting these animals in images, drawing bounding boxes around them for clear visualization.

## Features
- **High Accuracy**: Detects Striped Hyenas and Fishing Cats with 98.72% accuracy.
- **Real-Time Detection**: Processes images and provides immediate feedback on animal identification.
- **User-Friendly Interface**: Built with HTML, ensuring an intuitive user experience for uploading images and viewing results.
- **FastAPI Backend**: Efficiently handles image processing and model inference.

## Technologies Used
- **Machine Learning**: YOLOv8 for object detection
- **Frontend**: HTML/CSS for a responsive user interface
- **Backend**: FastAPI for handling API requests and model inference
- **Database**: (If applicable, mention the database used for storing results)

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aarzoo04/Wildlife-Species-Identification.git
   cd Wildlife-Species-Identification/api
2. **Set up a virtual environment (Optional)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
4. **Run the FastAPI server**:
   ```bash
   uvicorn main:app --reload
5. **Access the application**:
   ```bash
   Open index.html (from frontend folder) to use the application
## Usage
- Upload an image containing a Striped Hyena or Fishing Cat.
- Click the "Detect" button to process the image.
- View the results with bounding boxes indicating detected animals.
## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.
## Acknowledgements

- **YOLOv8** for the object detection model.
- **FastAPI** for creating the backend application.
