import cv2

# Load the video file
video = cv2.VideoCapture(0)

# get the frame rate of the video
frame_rate = video.get(cv2.CAP_PROP_FPS)
print("Frame rate of the video is: ", frame_rate)

# get the width and height of the video
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Width of the video is: ", width)
print("Height of the video is: ", height)

# create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, frame_rate, (width, height))


# Check if the file is opened successfully
if not video.isOpened():
    print("Error opening video stream or file")
    exit()



while True:
    # Read the first frame
    ret, frame = video.read()

    # Check if frame is not read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # write the frame to the output file
    out.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    cv2.imwrite('frame.jpg', gray);

    # Display the frame
    cv2.imshow('Frame', gray)

    # Wait for key press for 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()

# Close all the windows
cv2.destroyAllWindows()