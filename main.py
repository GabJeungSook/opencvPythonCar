import cv2

# Create a VideoCapture object
video_path1 = 'vids/highway.mp4'
video_path2 = 'vids/los_angeles.mp4'
video_path3 = 'vids/video1.avi'
video_path4 = 'vids/video2.avi'

cap = cv2.VideoCapture(video_path1)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Load the pre-trained car detection classifier (Haar cascade)
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Read and display frames until the video ends or the user presses a key
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was read correctly, process it
    if ret:
        # Convert the frame to grayscale for car detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars in the frame
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw green rectangles around the detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add text label above the rectangle
            cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()

#read image
#image = cv2.imread("images/Elon.jpg", 0)

#edges = cv2.Canny(image, 33, 33)

#cv2.imshow("Image", image)

#cv2.waitKey(0)

#video capture

# cap = cv2.VideoCapture(0)  # Open default camera
#
# while True:
#     ret, frame = cap.read()  # Read frame from camera
#
#     if not ret:
#         break
#
#     cv2.imshow('Camera Feed', frame)  # Display the frame
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' press
#         break
#
# cap.release()  # Release the camera
# cv2.destroyAllWindows()  # Close all windows