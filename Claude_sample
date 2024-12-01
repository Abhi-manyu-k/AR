import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class ARTrainingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Training steps for pump maintenance
        self.training_steps = [
            {
                "title": "Check Pump Pressure",
                "instruction": "Locate the pressure gauge and verify reading is within 30-50 PSI",
                "completion_criteria": "Press SPACE when pressure is verified",
                "ar_overlay": "pressure_gauge.obj"
            },
            {
                "title": "Inspect Seals",
                "instruction": "Check for any visible leaks around pump seals",
                "completion_criteria": "Press SPACE if no leaks, L if leaks detected",
                "ar_overlay": "seals_highlight.obj"
            },
            {
                "title": "Belt Tension",
                "instruction": "Verify belt deflection is 1/4 inch when pressed",
                "completion_criteria": "Press SPACE when verified, T to adjust tension",
                "ar_overlay": "belt_guide.obj"
            }
        ]
        self.current_step = 0
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrate_camera()

    def calibrate_camera(self):
        """Perform basic camera calibration."""
        ret, frame = self.cap.read()
        if ret:
            height, width = frame.shape[:2]
            focal_length = width
            center = (width/2, height/2)
            self.camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float32
            )
            self.dist_coeffs = np.zeros((4,1))

    def detect_markers(self, frame):
        """Detect ArUco markers in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids

    def render_ar_overlay(self, frame, corners, ids):
        """Render AR overlay based on detected markers."""
        if ids is not None:
            # Draw marker borders
            cv2.aruco.drawDetectedMarkers(frame, corners)
            
            # Get current training step
            step = self.training_steps[self.current_step]
            
            # Calculate marker pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.05, self.camera_matrix, self.dist_coeffs
            )
            
            # Draw 3D overlay (simplified for example)
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, 
                                rvecs[i], tvecs[i], 0.03)
                
                # Add step instructions
                cv2.putText(frame, f"Step {self.current_step + 1}: {step['title']}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, step['instruction'], 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, step['completion_criteria'], 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    def handle_user_input(self, key):
        """Handle user input for training progression."""
        if key == ord(' '):  # Space key
            if self.current_step < len(self.training_steps) - 1:
                self.current_step += 1
                print(f"\nProceeding to step {self.current_step + 1}")
            else:
                print("\nTraining completed!")
                return False
        elif key == ord('q'):  # Quit
            return False
        return True

    def run(self):
        """Main application loop."""
        print("AR Training Started - Press SPACE to advance steps, Q to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            corners, ids = self.detect_markers(frame)
            self.render_ar_overlay(frame, corners, ids)
            
            cv2.imshow('AR Training', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if not self.handle_user_input(key):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    app = ARTrainingApp()
    app.run()

if __name__ == "__main__":
    main()
