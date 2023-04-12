import numpy as np
import cv2 as cv


class Stabilizer:
    def __init__(self, video_in):
        self.input = self.connectVideo(video_in)
        self.transforms = []
        self.prev_center = None

    def connectVideo(self, video_in):
        return cv.VideoCapture(video_in)

    def getFace(self, frame):
        face_cascade = cv.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def getCenter(self, faces):
        for (x, y, w, h) in faces:
            center = (int(x + w/2), int(y + h/2))
        return center

    def drawCenter(self, frame, center):
        cv.circle(frame, center, 5, (255, 0, 0), -1)

    def getFeatures(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray, 200, 0.01, 10)
        return corners

    def getFlow(self, frame1, frame2):
        prev_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        prev_points = cv.goodFeaturesToTrack(prev_gray, 200, 0.01, 10)
        gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        points, status, error = cv.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_points, None)
        idx = np.where(status == 1)[0]
        prev_points = prev_points[idx]
        points = points[idx]

        return prev_points, points

    def getTransform(self, prev_points, points):
        m = cv.estimateAffine2D(prev_points, points)[0]
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        return dx, dy, da

    def trajectory(self):
        return np.cumsum(self.transforms, axis=0)

    def movingAverage(self, curve, radius):
        window_size = 2 * radius + 1
        f = np.ones(window_size)/window_size
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        curve_smoothed = curve_smoothed[radius:-radius]

        return curve_smoothed

    def smooth(self, trajectory):
        smoothed_trajectory = np.copy(trajectory)
        # Filter the x, y and angle curves
        for i in range(3):
            smoothed_trajectory[:, i] = self.movingAverage(
                trajectory[:, i], radius=30)

        return smoothed_trajectory

    def smoothTransforms(self, trajectory, smoothed_trajectory):
        difference = smoothed_trajectory - trajectory
        transforms_smooth = self.transforms + difference

        return transforms_smooth

    def remove_green(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([70, 255, 255])
        mask = cv.inRange(hsv, lower_green, upper_green)
        mask_inv = cv.bitwise_not(mask)

        fg = cv.bitwise_and(frame, frame, mask=mask_inv)
        bg = np.zeros_like(frame, np.uint8)

        return cv.bitwise_or(fg, bg)

    def stabilize(self):
        first_frame = True
        while True:
            ret, frame = self.input.read()
            if not ret:
                break

            if first_frame:
                prev_frame = frame.copy()
                prev_points = self.getFeatures(prev_frame)
                first_frame = False
                continue

            # Get face and center coordinates
            faces = self.getFace(frame)
            center = self.getCenter(faces)

            points = self.getFeatures(frame)

            # Get optical flow
            prev_points, points = self.getFlow(prev_frame, frame)

            # Get transform matrix
            dx, dy, da = self.getTransform(prev_points, points)

            # Store transform matrix
            self.transforms.append([dx, dy, da])
            if len(self.transforms) > 360:
                self.transforms.pop(0)

            # Get trajectory
            trajectory = self.trajectory()

            # Smooth trajectory
            smoothed_trajectory = self.smooth(trajectory)

            # Smooth transforms
            transforms_smooth = self.smoothTransforms(
                trajectory, smoothed_trajectory)

            # Apply transformation to frame
            M = np.zeros((2, 3), np.float32)
            M[0, 0] = np.cos(transforms_smooth[-1][2])
            M[0, 1] = -np.sin(transforms_smooth[-1][2])
            M[1, 0] = np.sin(transforms_smooth[-1][2])
            M[1, 1] = np.cos(transforms_smooth[-1][2])
            M[0, 2] = transforms_smooth[-1][0]
            M[1, 2] = transforms_smooth[-1][1]

            frame_stabilized = cv.warpAffine(
                frame, M, (frame.shape[1], frame.shape[0]))

            # Update previous frame
            prev_frame = frame.copy()
            prev_points = points.copy()

            # Draw face center on original frame
            self.drawCenter(frame, center)

            # Draw face center on stabilized frame
            newfaces = self.getFace(frame_stabilized)
            if len(newfaces) > 0:
                newcenter = self.getCenter(newfaces)
            self.drawCenter(frame_stabilized, newcenter)

            # Remove green background
            frame_stabilized = self.remove_green(frame_stabilized)
            frame = self.remove_green(frame)

            # Display original and stabilized video streams side by side
            out = np.hstack((frame, frame_stabilized))
            cv.imshow('Unstabilised {} and {} Stabilized'.format(
                " "*40, " "*40), out)

            # Wait for key press to exit
            if cv.waitKey(1) == ord('q'):
                break

        # Release resources and destroy windows
        self.input.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    video_in = 'walkingCropNonStab.mp4'
    stabilizer = Stabilizer(video_in)
    stabilizer.stabilize()
