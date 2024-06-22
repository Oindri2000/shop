#=============================import required libraries================================================
from ultralytics import YOLO  # Import the YOLO model from the ultralytics library
import cv2  # Import OpenCV for image and video processing
import easyocr  # Import EasyOCR for Optical Character Recognition (OCR)
import numpy as np  # Import NumPy for numerical operations
import warnings  # Import warnings to handle warning messages
import csv  # Import CSV to handle CSV file operations

# Ignore all warnings
warnings.filterwarnings('ignore')

#====================================shop model and ocr ================================================

# Load my YOLO model
model = YOLO('house_best.pt')
# Initialize EasyOCR reader with English language
reader = easyocr.Reader(['en'])

#===========================================load video================================================

# Define the path to the input video
video_path = '/home/oindriaiml/Downloads/13948918-hd_1920_1080_60fps (1).mp4'
# Define the path to the output video
output_path = 'updated_output_video3.avi'
# Open the video file
cap = cv2.VideoCapture(video_path)
# Define the codec for the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Get the frames per second (FPS) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
# Get the total number of frames in the input video
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(total_frame)
# Create a VideoWriter object to save the output video
out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
# Initialize frame count
frame_count = 0

# Initialize CSV file
csv_file_path = 'ocr_results_3.csv'
# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row in the CSV file
    writer.writerow(["Frame Count", "Timestamp", "OCR Result"])

    # Commented out the preprocessing function for now
    # def preprocess_image(image):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    #     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply thresholding
    #     denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)  # Apply denoising
    #     sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #     sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)  # Apply sharpening
    #     return sharpened

    # Loop through each frame of the video
    while cap.isOpened():
        frame_count += 1
        print(frame_count)
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        #========================================Result findings================================================

        # Predict objects in the frame using the YOLO model
        result = model.predict(frame, conf=0.1, iou=0.1)  #  we want all classes

        # Loop through each detection in the result
        for detection in result[0].boxes:
            x1, y1, x2, y2 = detection.xyxy.tolist()[0]  # Get the coordinates of the detection
            cls = int(detection.cls)  # Get the class of the detection
            # Crop the detected object from the frame
            image_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Apply OCR only to class 1 (assuming class 1 is the shop name)
            if cls == 1:
                try:
                    # Preprocess the cropped image before OCR (if needed)
                    #preprocessed_image = preprocess_image(image_crop)
                    # Perform OCR on the cropped image
                    result_texts = reader.readtext(image_crop, detail=0)
                    # Get the first OCR result or default to "shop_name" if no text is found
                    if result_texts:
                        result_text = result_texts[0]
                    else:
                        result_text = "shop_name"
                    
                    # Highlight where the OCR is working and result is detected
                    print(f"OCR Result for class 1: {result_text}")

                    # Calculate the timestamp for the current frame
                    timestamp = frame_count / fps

                    # Highlight where the OCR and YOLO result (shop_name) is processed in every frame
                    # Append the OCR result and timestamp to the CSV file
                    writer.writerow([frame_count, timestamp, result_text])

                    # Display the OCR result on the frame
                    (label_width, label_height), baseline = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    # Draw a rectangle to display the OCR text
                    cv2.rectangle(frame, (int(x1), int(y1) - label_height - baseline), (int(x1) + label_width, int(y1)), (0, 255, 0), cv2.FILLED)
                    # Draw a rectangle around the detected object
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Put the OCR text on the frame
                    cv2.putText(frame, result_text, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                except Exception as e:
                    print(f"Error during OCR: {e}")

        # Write the processed frame into the output video
        out.write(frame)

# Release everything once the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
