import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import time

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Landmark indices
        # Left eye (Subject's Left, Image Right)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 374]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.LEFT_PUPIL = 473
        
        # Right eye (Subject's Right, Image Left)
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.RIGHT_PUPIL = 468

    def get_gaze_ratio(self, eye_points, pupil_point, frame_w, frame_h):
        # Extract coordinates
        eye_coords = np.array([(p.x * frame_w, p.y * frame_h) for p in eye_points])
        pupil_coord = np.array([pupil_point.x * frame_w, pupil_point.y * frame_h])
        
        # Get bounds
        left_bound = np.min(eye_coords[:, 0]) # Image Left
        right_bound = np.max(eye_coords[:, 0]) # Image Right
        top_bound = np.min(eye_coords[:, 1])
        bottom_bound = np.max(eye_coords[:, 1])
        
        eye_width = right_bound - left_bound
        eye_height = bottom_bound - top_bound
        
        if eye_width == 0 or eye_height == 0:
            return 0.5, 0.5

        # Horizontal ratio (0.0 = Left/Inner, 1.0 = Right/Outer in image coordinates)
        # For subject's right eye (Image Left):
        # 0.0 means looking to their right (Image Left)
        # 1.0 means looking to their left (Image Right)
        h_ratio = (pupil_coord[0] - left_bound) / eye_width
        v_ratio = (pupil_coord[1] - top_bound) / eye_height
        
        return h_ratio, v_ratio

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        gaze_status = "UNKNOWN"
        gaze_vector = (0, 0)
        landmarks_detected = False
        
        if results.multi_face_landmarks:
            landmarks_detected = True
            mesh_points = results.multi_face_landmarks[0].landmark
            
            # Helper to get point
            def get_point(idx): return mesh_points[idx]
            
            # Right Eye (Image Left)
            r_eye_pts = [get_point(i) for i in self.RIGHT_EYE]
            r_pupil = get_point(self.RIGHT_PUPIL)
            rh, rv = self.get_gaze_ratio(r_eye_pts, r_pupil, w, h)
            
            # Left Eye (Image Right)
            l_eye_pts = [get_point(i) for i in self.LEFT_EYE]
            l_pupil = get_point(self.LEFT_PUPIL)
            lh, lv = self.get_gaze_ratio(l_eye_pts, l_pupil, w, h)
            
            # Average ratio
            avg_h = (rh + lh) / 2
            avg_v = (rv + lv) / 2
            gaze_vector = (avg_h, avg_v)

            # Draw Eyes (Optional: Draw contours)
            for point in r_eye_pts + l_eye_pts:
                cv2.circle(frame, (int(point.x * w), int(point.y * h)), 1, (0, 255, 0), -1)
            
            # Draw Iris
            cv2.circle(frame, (int(r_pupil.x * w), int(r_pupil.y * h)), 2, (0, 0, 255), -1)
            cv2.circle(frame, (int(l_pupil.x * w), int(l_pupil.y * h)), 2, (0, 0, 255), -1)

            # Determine Direction
            # Thresholds need tuning based on camera angle
            # Assuming frontal/near-frontal
            if avg_h < 0.35:
                direction_h = "RIGHT" # Image Left (Subject's Right)
            elif avg_h > 0.65:
                direction_h = "LEFT" # Image Right (Subject's Left)
            else:
                direction_h = "CENTER"
                
            if avg_v < 0.35: # Up
                direction_v = "UP"
            elif avg_v > 0.65: # Down (Dashboard/Wheel?)
                direction_v = "DOWN" 
            else:
                direction_v = "CENTER"
                
            gaze_status = f"{direction_h}-{direction_v}"
            
            # Specialized detection for driving
            if direction_v == "DOWN":
                gaze_status += " (Checking Dashboard?)"
            elif direction_h == "RIGHT": 
                gaze_status += " (Side Mirror?)"

        return frame, gaze_status, gaze_vector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="output/gaze_output.mp4", help="Path to output video")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error opening video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    tracker = GazeTracker()
    
    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, status, vector = tracker.process_frame(frame)

        # Overlay Info
        cv2.putText(processed_frame, f"Gaze: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Vector: ({vector[0]:.2f}, {vector[1]:.2f})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        writer.write(processed_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

    t_end = time.time()
    print(f"\nDone. Processed {frame_count} frames in {t_end - t_start:.2f}s")
    
    cap.release()
    writer.release()
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
