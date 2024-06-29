import cv2
import numpy as np
import time
import pickle
import face_recognition
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

def capture_attendance():
    """
    Capture attendance using face recognition.

    This function captures attendance by comparing the faces in the video stream with the known faces.
    It uses the face_recognition library to detect and encode faces, and compares them with the known faces
    stored in the 'EncodeFile.p' file. The attendance is marked in a CSV file named 'attendance.csv'.

    Returns:
        None
    """

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    file = open("EncodeFile.p", 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, studentIds = encodeListKnownWithIds

    # Read the existing CSV file into a DataFrame
    try:
        df = pd.read_csv('attendance.csv')
        df['StudentID'] = df['StudentID'].astype(str)  # convert student IDs to strings
    except pd.errors.EmptyDataError:
        # If the CSV file is empty, create a new DataFrame
        df = pd.DataFrame(columns=['StudentID'])
        df['StudentID'] = studentIds
        df = df.fillna('Absent')

    # Get the current date as a string
    date = datetime.now().strftime('%Y-%m-%d %H:%M')

    # If the current date is not already a column in the DataFrame, add it
    if date not in df.columns:
        df[date] = 'Absent'  # default to 'Absent' for all students

    start_time = time.time()
    marked_students = []  # list of students who have already been marked present

    # Performance metrics
    total_frames = 0
    marked_present = 0
    marked_absent = 0

    accuracy_values = []  # list to store accuracy values over time

    while True:
        if time.time() - start_time >= 1 * 60:
            # If 45 minutes have passed, create a new attendance column for the current time
            date = datetime.now().strftime('%Y-%m-%d %H:%M')
            df[date] = 'Absent'
            start_time = time.time()  # reset the start time
            marked_students = []  # reset the list of marked students

            # Update performance metrics
            total_frames = 0
            marked_present = 0
            marked_absent = 0

        img = cap.read()[1]

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            # Check if the student has already been marked present
            if studentIds[matchIndex] not in marked_students:
                # If not, mark them as present and add them to the list of marked students
                df.loc[df['StudentID'] == studentIds[matchIndex], date] = 'Present'
                marked_students.append(studentIds[matchIndex])
                print("Student ID: ", studentIds[matchIndex], " marked as present")
                marked_present += 1
            else:
                print("Student ID: ", studentIds[matchIndex], " has already been marked as present")
                marked_absent += 1

            # Draw a rectangle around the face and put a label with the student ID
            top, right, bottom, left = faceLoc
            top, right, bottom, left = top*4, right*4, bottom*4, left*4  # Scale back up face locations since the frame was scaled to 1/4 size
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, studentIds[matchIndex], (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam', img)  # Display the image with the marked faces
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
        total_frames += 1

        df.to_csv('attendance.csv', index=False)  # save the updated DataFrame to the CSV file

        # Update accuracy values
        accuracy = marked_present / total_frames
        accuracy_values.append(accuracy)

        # Print performance metrics
        print("Total Frames:", total_frames)
        print("Marked Present:", marked_present)
        print("Marked Absent:", marked_absent)

    # Plot accuracy over time
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(accuracy_values)), accuracy_values)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('Face Recognition Accuracy Over Time')
    plt.savefig('Results\accuracy_over_time.png')  # Save the graph as an image
    plt.show()

    # Plot the stored encodings
    plt.figure(figsize=(10, 8))
    sns.heatmap(encodeListKnown, cmap='viridis')
    plt.title('Stored Encodings')
    plt.savefig('Results\Stored_encoding_heatmap.png')  # Save the graph as an image
    plt.show()

    # Plot the detected image encodings
    plt.figure(figsize=(10, 8))
    sns.heatmap(encodeCurFrame, cmap='viridis')
    plt.title('Detected Image Encodings')
    plt.savefig('Results\Detected_encoding_heatmap.png')
    plt.show()

    # Calculate and plot the distances between the stored encodings and the detected image encodings
    distances = np.zeros((len(encodeListKnown), len(encodeCurFrame)))
    for i, encodeKnown in enumerate(encodeListKnown):
        for j, encodeCur in enumerate(encodeCurFrame):
            distances[i, j] = np.linalg.norm(np.array(encodeKnown) - np.array(encodeCur))
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, cmap='viridis')
    plt.title('Distances Between Stored Encodings and Detected Image Encodings')
    plt.savefig('Results\Distances_heatmap.png')
    plt.show()


    cap.release()
    cv2.destroyAllWindows()


capture_attendance()
