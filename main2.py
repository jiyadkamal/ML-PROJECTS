import cv2
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for efficient text similarity

# Function to compare texts using Sentence Transformers
def compare_texts_with_transformer(text1, text2, threshold=0.85):
    # Encode the texts into embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    # Return 'yes' if similarity is above the threshold, otherwise 'no'
    return "yes" if similarity >= threshold else "no"

# Function to extract text from video and compare frames
def extract_text_from_video(video_path, skip_frames=5):
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if video is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    previous_text = None
    frame_counter = 0  # Frame counter to skip frames

    # Loop through each frame in the video
    while True:
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            break

        # Skip frames based on the skip_frames parameter
        if frame_counter % skip_frames != 0:
            frame_counter += 1
            continue

        # Convert the frame to a PIL Image (needed for pytesseract)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Use pytesseract to extract text from the image (frame)
        current_text = pytesseract.image_to_string(pil_image).strip()

        # Perform OCR on the first frame and whenever the texts are different
        if previous_text is None:
            # Always do OCR on the first frame
            print("Extracted Text from first frame:")
            print(current_text)
        else:
            # Call the Sentence Transformer to compare the current and previous texts
            result = compare_texts_with_transformer(previous_text, current_text)
            print(f"Comparison result: {result}")

            # If the texts are not similar, perform OCR
            if result.lower() == "no":
                print("Extracted Text from current frame:")
                print(current_text)
            else:
                print("Skipping OCR for this frame (texts are similar).")

        # Store the current text as previous text for the next comparison
        previous_text = current_text

        # Optional: You can display the video frame (press 'q' to exit the window)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Increment frame counter
        frame_counter += 1

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
video_path = 'text2.mp4'  # Replace with the path to your video file
skip_frames = 20  # Set to 5 to skip 5 frames between each OCR process
extract_text_from_video(video_path, skip_frames)
