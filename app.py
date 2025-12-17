import streamlit as st
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils

# Page Setup
st.set_page_config(page_title="AI OMR Scanner 2025", layout="wide")
st.title("🎯 Modern AI OMR Scanner")
st.write("Upload a photo of a 5-question bubble sheet (A-E) to see the magic.")

# Sidebar for Answer Key
st.sidebar.header("Configuration")
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1} # Q1:B, Q2:E, Q3:A, Q4:D, Q5:B
st.sidebar.write("Answer Key Set:", ANSWER_KEY)

uploaded_file = st.file_uploader("Upload OMR Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. Image Loading
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 2. Find the Paper (Perspective Transform)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is not None:
        paper = four_point_transform(image, docCnt.reshape(4, 2))
        warped = four_point_transform(gray, docCnt.reshape(4, 2))
        
        # 3. Process Bubbles
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
                questionCnts.append(c)

        # Sort questions top-to-bottom
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        correct = 0

        # 4. Grading Logic
        for (q, i) in enumerate(range(0, len(questionCnts), 5)):
            cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
            bubbled = None

            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)

                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)

            color = (0, 0, 255)
            k = ANSWER_KEY[q]
            if k == bubbled[1]:
                color = (0, 255, 0)
                correct += 1
            cv2.drawContours(paper, [cnts[k]], -1, color, 3)

        # 5. Show Results
        score = (correct / 5.0) * 100
        st.subheader(f"Final Score: {score}%")
        st.image(paper, caption="Graded OMR Sheet", use_container_width=True)
    else:
        st.error("Could not find the OMR sheet in the photo. Try a clearer background.")
