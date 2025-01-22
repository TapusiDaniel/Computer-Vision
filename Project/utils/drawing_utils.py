import cv2
import numpy as np

def draw_boxes(image, detections, colors):
    """Draw boxes around detected objects with team colors."""
    image_copy = image.copy()
    
    # Define team colors if not in config
    team_colors = {
        'team1': (0, 0, 255),    # Red
        'team2': (255, 0, 0),    # Blue
        'referee': (0, 0, 0),    # Black
        'ball': (0, 255, 255)    # Yellow
    }
    team_colors.update(colors)

    for det in detections:
        bbox = det['bbox']
        class_id = det.get('class_id')
        confidence = det.get('confidence', 0)
        team = det.get('team', 'unknown')
        is_goalkeeper = det.get('is_goalkeeper', False)

        # Choose color based on team or object type
        if class_id == 0:  # Person
            color = team_colors.get(team, (128, 128, 128))
            # Add different border style for goalkeeper
            thickness = 3 if is_goalkeeper else 2
            box_style = cv2.LINE_AA if is_goalkeeper else cv2.LINE_8
        elif class_id == 32:  # Ball
            color = team_colors['ball']
            thickness = 2
            box_style = cv2.LINE_8
        else:
            continue

        # Draw bounding box
        cv2.rectangle(image_copy, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     color, thickness, box_style)

        # Draw label
        label = f"{team}"
        if is_goalkeeper:
            label += " (GK)"
        label += f" {confidence:.2f}"
        
        label_size, baseline = cv2.getTextSize(label, 
                                             cv2.FONT_HERSHEY_SIMPLEX,
                                             0.5, thickness)
        y = bbox[1] - 15 if bbox[1] - 15 > 15 else bbox[1] + 15
        cv2.putText(image_copy, label,
                    (int(bbox[0]), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, thickness)

    return image_copy