import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Set up the Streamlit page layout
st.set_page_config(layout="wide")
st.image(r"math.jpeg")

col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Initialize Gemini AI API (DO NOT expose your key publicly)
genai.configure(api_key="AIzaSyCfnqmV_Pds6MH77kaD07jLzZDGYkCE7Xc")  # Replace with a secure method
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam
cap = cv2.VideoCapture(0)  # Changed to default camera (1 might not be valid)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize HandDetector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    """Detects hand landmarks and fingers."""
    if img is None or img.size == 0:
        print("Warning: Image is empty or invalid.")
        return None

    hands, img = detector.findHands(img, draw=False, flipType=True)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None


def draw(info, prev_pos, canvas, img):
    """Draws a line based on finger positions."""
    if info is None:
        return prev_pos, canvas

    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # Index finger is up
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 0, 255), 10)

    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up clears canvas
        canvas = np.zeros_like(img)

    return current_pos, canvas


def sendToAI(model, canvas, fingers):
    """Sends the drawing to the AI model when all fingers except the pinky are up."""
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""


prev_pos = None
canvas = None
output_text = ""

# **Avoid infinite loop in Streamlit**
while run:
    success, img = cap.read()
    if not success or img is None:
        st.warning("Failed to capture webcam image.")
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas, img)
        output_text = sendToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    # Exit condition for webcam (Press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
