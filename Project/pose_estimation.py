import cv2
import numpy as np
import math

class PoseEstimator:
    def __init__(self):
        self.keypoints = ['rightShoulder', 'leftShoulder', 'leftHip', 'rightHip',
                         'rightKnee', 'leftKnee', 'rightAnkle', 'leftAnkle']
        
    def estimate(self, image, bbox):
        """Get pose estimation with keypoints"""
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        width = x2 - x1
        
        # Placeholder keypoint detection - replace with actual pose estimation
        keypoints = {}
        for part in self.keypoints:
            if 'Ankle' in part:
                y_offset = 0.95  # Ankles near bottom
            elif 'Knee' in part:
                y_offset = 0.7   # Knees in middle-lower
            elif 'Hip' in part:
                y_offset = 0.5   # Hips in middle
            else:  # Shoulders
                y_offset = 0.2   # Shoulders near top
                
            keypoints[part] = {
                'x': x1 + (width * (0.3 if 'left' in part.lower() else 0.7)),
                'y': y1 + (height * y_offset),
                'confidence': 0.8
            }
        
        return {
            'bbox': bbox,
            'keypoints': keypoints,
            'leftmost_point': [x1 + width/2, y2]  # Default to bottom center
        }

def get_angle(vanishing_point, point, image, goal_direction):
    """Calculate angle between point and vanishing point"""
    if goal_direction == 'right':
        reference_point = [0, vanishing_point[1]]
    else:
        reference_point = [image.shape[1], vanishing_point[1]]
    
    ba = np.array(reference_point) - np.array(vanishing_point)
    bc = np.array(point) - np.array(vanishing_point)
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    # Reverse the comparison logic for left-to-right vs right-to-left
    if goal_direction == 'left':
        # For left direction, we want the smallest x-coordinate (leftmost point)
        angle = -angle if point[0] < vanishing_point[0] else angle
    else:
        # For right direction, we want the largest x-coordinate (rightmost point)
        angle = -angle if point[0] > vanishing_point[0] else angle
            
    return angle

def project_point_on_plane(plane, point, pose, ratio, image):
    """
    Project point onto ground plane using player height estimation
    """
    a, b, c, d = plane
    x1, y1 = point
    keypoints = pose['keypoints']
    
    # Calculate body height using shoulder-hip distance if available
    if all(k in keypoints for k in ['rightShoulder', 'rightHip', 'leftShoulder', 'leftHip']):
        right_upper_body = math.sqrt(
            (keypoints['rightShoulder']['y'] - keypoints['rightHip']['y']) ** 2 +
            (keypoints['rightShoulder']['x'] - keypoints['rightHip']['x']) ** 2
        )
        left_upper_body = math.sqrt(
            (keypoints['leftShoulder']['y'] - keypoints['leftHip']['y']) ** 2 +
            (keypoints['leftShoulder']['x'] - keypoints['leftHip']['x']) ** 2
        )
        height = ratio * ((right_upper_body + left_upper_body)/2)
    else:
        # Fallback to using box height
        bbox = pose['bbox']
        height = ratio * (bbox[3] - bbox[1])
    
    # Project point
    z1 = height
    perp_dist = abs((a * x1 + b * y1 + c * z1 + d)) / math.sqrt(a * a + b * b + c * c)
    y_new = y1 + perp_dist
    
    return [int(x1), int(y_new)]

def get_leftmost_point(detection, vertical_vanishing_point, ground_plane, image, goal_direction='left'):
    """
    Find leftmost point using the bottom-left corner of bounding box
    """
    # Get bounding box coordinates
    x1, y1, x2, y2 = detection['bbox']
    
    # Use the bottom-left corner of the bounding box
    # This is where the player touches the ground
    leftmost_point = [int(x1), int(y2)]
    
    return leftmost_point