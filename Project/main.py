import cv2
import numpy as np
import sys
from utils.drawing_utils import draw_boxes
from offside_decision import adjust_for_perspective, get_player_positions, find_last_defender, analyze_offside
from box_detection import detect_objects
from team_classifier import TeamClassifier
from pose_estimation import (
    PoseEstimator, get_angle, project_point_on_plane, get_leftmost_point
)
from line_detection import (
    detect_field_lines, get_vertical_vanishing_point, get_horizontal_vanishing_point,
    print_lines_and_angles, filter_negative_angles, group_lines_by_angle,
    filter_clusters, visualize_vanishing_point, get_ground_plane
)

def process_offside(image, detections, field_dimensions=None):
    """
    Main function to process offside situation with perspective correction
    """
    try:
        height, width = image.shape[:2]
        
        # Add perspective-adjusted positions to detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            foot_position = ((x1 + x2) // 2, y2)
            detection['projected_leftmost_point'] = foot_position
        
        # Find last defender with perspective correction
        defender_pos = find_last_defender(detections)
        if defender_pos is None:
            return image, {'error': 'No defender found'}
            
        # Analyze offside with perspective correction
        annotated_image, offside_players = analyze_offside(
            image, 
            detections,
            field_dimensions=field_dimensions
        )
        
        offside_result = {
            'offside_detected': len(offside_players) > 0,
            'num_offside_players': len(offside_players),
            'offside_positions': offside_players,
            'last_defender_pos': defender_pos
        }
        
        return annotated_image, offside_result
        
    except Exception as e:
        print(f"Error in offside processing: {e}")
        import traceback
        print(traceback.format_exc())
        return image, None

def main():
    # Define team colors
    colors = {
        'team1': (0, 0, 255),    # Red
        'team2': (255, 0, 0),    # Blue
        'referee': (0, 0, 0),    # Black
        'ball': (0, 255, 255)    # Yellow
    }

    # Initialize team classifier
    team_classifier = TeamClassifier()
    
    # Load image
    image = cv2.imread('374.jpg')
    if image is None:
        print("Error: Image not found. Please check the file path.")
        return

    # Process player detections first
    detections = detect_objects(image)
    detections = team_classifier.cluster_teams(image, detections)
    
    # Add projected leftmost point to each detection
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        # Use the leftmost point of the bottom edge
        foot_position = (int(x1), int(y2)) 
        detection['projected_leftmost_point'] = foot_position
    
    # Print player positions for debugging
    print("\nPlayer Positions:")
    for i, detection in enumerate(detections):
        team = detection.get('team', 'unknown')
        x, y = detection['projected_leftmost_point']
        print(f"Player {i + 1}: Team = {team}, Position = ({x}, {y})")
    
    player_image = draw_boxes(image.copy(), detections, colors)
    
    # Save player detection image
    cv2.imwrite('output_players.jpg', player_image)
    print("Saved output_players.jpg")

    # Detect field lines
    lines = detect_field_lines(image)
    
    if lines is None:
        print("No lines detected.")
        return
        
    # Get vertical lines for vanishing point calculation
    vertical_lines = filter_negative_angles(lines)
    
    # Print lines and their angles
    print_lines_and_angles(vertical_lines)
    
    # Group lines by angle
    clusters = group_lines_by_angle(vertical_lines)
    
    # Filter clusters to keep only those with at least 5 lines
    filtered_clusters = filter_clusters(clusters, min_lines=5)
    
    # Get vertical vanishing point
    vertical_vanishing_point = get_vertical_vanishing_point(vertical_lines)
    if vertical_vanishing_point is None:
        print("No vertical vanishing point found.")
        return

    # Visualize vanishing point and field lines
    visualization = visualize_vanishing_point(
        image, 
        filtered_clusters, 
        vertical_vanishing_point,
        'output_vanishing.jpg',
        show_window=False,
        detections=detections
    )
    print("Saved output_vanishing.jpg")
    print("Vertical vanishing point:", vertical_vanishing_point)

    # Process offside detection using the SAME vanishing point and field lines
    offside_canvas, num_offside = analyze_offside(
        image, 
        detections,
        vertical_vanishing_point,  # Use the precomputed vanishing point
        filtered_clusters          # Use the precomputed field lines
    )

    # Save offside analysis image
    cv2.imwrite('output_offside.jpg', offside_canvas)
    print("Saved output_offside.jpg with goalkeeper's line in yellow.")

    # Debug: Check if the file exists
    import os
    if os.path.exists('output_offside.jpg'):
        print("output_offside.jpg exists and was saved successfully.")
    else:
        print("Error: output_offside.jpg was not saved.")
    print(f"\nOffside Analysis Results:")
    print(f"Number of players in offside position: {num_offside}")
    
    # Show final results
    cv2.imshow('Offside Analysis', offside_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)