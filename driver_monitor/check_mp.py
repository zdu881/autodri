import mediapipe as mp
print("MediaPipe file:", mp.__file__)
try:
    print("Solutions:", mp.solutions)
except AttributeError as e:
    print("Error:", e)
