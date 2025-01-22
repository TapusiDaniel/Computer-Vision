import cv2
import numpy as np
from line_detection import visualize_vanishing_point

def adjust_for_perspective(point, image_height):
    """
    Adjust coordinates to account for perspective distortion
    """
    x, y = point
    # Weight based on vertical position in image
    perspective_weight = (y / image_height)
    adjusted_x = x * perspective_weight
    return (adjusted_x, y)

def get_player_positions(detections, team):
    """
    Get positions of players from a specific team
    
    Args:
        detections: List of player detections with bbox and team information
        team: Team identifier ('team1' or 'team2')
    
    Returns:
        List of tuples (x, y) representing player foot positions
    """
    positions = []
    for detection in detections:
        if detection.get('team') == team:
            x1, y1, x2, y2 = detection['bbox']
            # Use bottom center of bounding box as player position
            foot_position = ((x1 + x2) // 2, y2)
            positions.append(foot_position)
    return positions

def find_last_defender(detections, defending_team='team1'):
    """
    Find the last defender's position with perspective adjustment, excluding goalkeeper
    """
    # Get all defenders (team1 players)
    defenders = [d for d in detections if d['team'] == defending_team]
    
    # Find goalkeeper (leftmost x-coordinate without perspective adjustment)
    goalkeeper_pos = min(defenders, key=lambda x: x['projected_leftmost_point'][0])
    
    # Remove goalkeeper from defenders list
    defenders = [d for d in defenders if d != goalkeeper_pos]
    
    # Adjust each defender's position for perspective
    adjusted_defenders = []
    for d in defenders:
        pos = d['projected_leftmost_point']
        # Invert the weight - players higher up (smaller y) get larger weight
        weight = 1 - (pos[1] / 1080)  # This will give higher weights to smaller y values
        adjusted_x = pos[0] * (1 + weight)  # Add 1 to keep x positive
        adjusted_defenders.append((adjusted_x, pos, d))
    
    # Sort by adjusted x-coordinate
    sorted_defenders = sorted(adjusted_defenders, key=lambda x: x[0])
    
    # Print all defender positions for debugging
    print("\nAll defenders sorted by perspective-adjusted x-coordinate (excluding goalkeeper):")
    for i, (adj_x, pos, d) in enumerate(sorted_defenders):
        print(f"Defender {i+1}: ({int(pos[0])}, {int(pos[1])}) - adjusted_x: {int(adj_x)}")
    
    if len(sorted_defenders) < 2:
        return None
        
    # Return original position of the leftmost defender
    return sorted_defenders[0][1]

def find_attackers_positions(detections):
    """
    Find all attacking players positions (team2)
    
    Args:
        detections: List of player detections
    
    Returns:
        List of (x, y) positions of attacking players
    """
    return get_player_positions(detections, 'team2')

def is_player_offside(player_pos, defender_pos, field_dimensions=None):
    """
    Check if a player is in offside position with perspective correction
    """
    player_x, player_y = player_pos
    defender_x, defender_y = defender_pos
    
    # Adjust positions based on y-coordinate (perspective)
    player_weight = (player_y / 1080)  # Normalize by image height
    defender_weight = (defender_y / 1080)
    
    adjusted_player_x = player_x * player_weight
    adjusted_defender_x = defender_x * defender_weight
    
    return adjusted_player_x < adjusted_defender_x

def draw_offside_line(image, defender_pos, vanishing_point):
    """
    Draw the offside line from defender through vanishing point
    """
    result = image.copy()
    
    if vanishing_point is not None:
        # Draw line from defender to vanishing point
        vx, vy = vanishing_point
        dx, dy = defender_pos
        cv2.line(result, 
                 (int(dx), int(dy)), 
                 (int(vx), int(vy)), 
                 (255, 255, 0), 2)  # Yellow line
    
    return result

def analyze_offside(image, detections, vertical_vanishing_point, filtered_clusters):
    """
    Analyze offside using precomputed vanishing point and field lines.
    """
    # Only get the offside visualization
    offside_canvas = visualize_vanishing_point(
        image, 
        filtered_clusters, 
        vertical_vanishing_point,
        'output_vanishing.jpg',
        show_window=False,
        detections=detections
    )

    return offside_canvas, 0

def process_offside(image, detections, vanishing_point=None):
    """
    Main function to process offside situation
    
    Args:
        image: Input image
        detections: List of player detections
        vanishing_point: Optional (x, y) vanishing point for perspective correction
    
    Returns:
        Tuple (annotated_image, offside_result) where offside_result is a dict
        containing analysis results
    """
    try:
        # Analyze offside situation
        annotated_image, offside_players = analyze_offside(image, detections, vanishing_point)
        
        # Prepare result
        offside_result = {
            'offside_detected': len(offside_players) > 0,
            'num_offside_players': len(offside_players),
            'offside_positions': offside_players
        }
        
        return annotated_image, offside_result
        
    except Exception as e:
        print(f"Error in offside processing: {e}")
        import traceback
        print(traceback.format_exc())
        return image, None