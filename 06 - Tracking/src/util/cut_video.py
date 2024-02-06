import cv2


def cut_and_save_video(input_video_path, output_video_path, starting_frame, ending_frame):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()

        # Break the loop if the video is over
        if not ret:
            break

        # Process frames within the specified range
        if starting_frame <= frame_count <= ending_frame:
            out.write(frame)

        frame_count += 1

    # Release video capture and writer objects
    cap.release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Specify the input video file path
    input_video_path = "data/video3.avi"

    # Specify the output video file path
    output_video_path = "data/cut_video3.avi"
    # Specify the range of frames to include [starting_frame, ending_frame]
    starting_frame = 1
    ending_frame = 60

    # Call the function to cut and save the video
    cut_and_save_video(input_video_path, output_video_path,
                       starting_frame, ending_frame)
