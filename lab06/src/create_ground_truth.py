import os
import cv2
from matplotlib import pyplot as plt

# Initialize variables
data_dir = './data/'
video_name = 'video1.avi'
frame_centers = []
box_size = 13  # Adjust the size of the box as needed
current_frame = 0
first_frame = 10

# Callback function for mouse events to get the center of the box


def mouse_callback(event):
    global frame_centers, current_frame
    if event.button == 1:
        frame_centers.append((int(event.xdata), int(event.ydata)))
        print("Frame:", current_frame, "Center:", frame_centers[current_frame])
    elif event.button == 3:
        frame_centers.pop()
        current_frame -= 1
        print("Frame:", current_frame, "Center:", frame_centers[current_frame])

# Function to draw the box on the frame


def draw_box(frame, top_left, bottom_right):
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)


# Open the video file
video_path = os.path.join(data_dir, video_name)
cap = cv2.VideoCapture(video_path)
cap.set(1, first_frame)
_, first_image = cap.read()

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Create a window and set the callback function
cv2.namedWindow('Video Player')
cv2.setMouseCallback('Video Player', mouse_callback)

# Create a figure and set the callback function
fig, ax = plt.subplots()
fig.canvas.mpl_connect('button_press_event', mouse_callback)

top_left = (50, 50)  # Initial position of the box
frame_centers.append(top_left)


while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no more frames are available

    # Draw the box on the frame
    # draw box with center at frame_centers[current_frame]
    draw_box(frame, (frame_centers[current_frame][0] - box_size // 2, frame_centers[current_frame][1] - box_size //
             2), (frame_centers[current_frame][0] + box_size // 2, frame_centers[current_frame][1] + box_size // 2))

    # Display the frame using Matplotlib
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.01)

    # Move to the next frame
    current_frame += 1

    plt.waitforbuttonpress()

# Release the video capture object
cap.release()

# Print the list of frame centers
print("Frame centers:", frame_centers)

# Close the Matplotlib window
plt.close()
