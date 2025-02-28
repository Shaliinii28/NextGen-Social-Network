import cv2
import face_recognition
import os
import base64

def FaceVerification(facecollection):
    known_face_names = []
    known_face_encodings = []

    # Load known face encodings and names
    for profile in facecollection:
        person = profile
        image_of_person = face_recognition.load_image_file(person)
        face_encodings = face_recognition.face_encodings(image_of_person)
        if len(face_encodings) > 0:
            person_face_encoding = face_encodings[0]  # Take the first face encoding
            known_face_encodings.append(person_face_encoding)
            known_face_names.append(os.path.basename(person))
        else:
            print(f"No face detected in {os.path.basename(person)}")

    # Create a VideoCapture object to capture the video from the camera
    video_capture = cv2.VideoCapture(0)

    # Read a single frame from the video stream
    ret, frame = video_capture.read()

    # Convert the frame to RGB for face recognition
    rgb_frame = frame[:, :, ::-1]

    # Detect faces in the RGB frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over the detected faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the face encoding with the known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the corresponding name
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            print(f"Match found: {name}")
            # Release the video capture object and close all windows
            video_capture.release()
            cv2.destroyAllWindows()
            return True,None

        # Draw a rectangle around the detected face and display the name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

    # Display the resulting frame

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()
    _, img_encoded = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(img_encoded).decode('utf-8')
    # If no match is found, return the current image from the camera
    return False,encoded_image
