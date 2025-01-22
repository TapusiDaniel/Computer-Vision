import cv2
import numpy as np

def detect_field_lines(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color (field)
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([86, 255, 255])
    
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    return lines

def calculate_line_angle(line):
    """
    Calculate the angle of a line in degrees.
    
    Args:
        line: A line represented as [x1, y1, x2, y2].
    
    Returns:
        angle: The angle of the line in degrees.
    """
    x1, y1, x2, y2 = line
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return angle

def filter_negative_angles(lines):
    """
    Filter out lines with non-negative angles.
    
    Args:
        lines: List of detected lines.
    
    Returns:
        negative_lines: List of lines with negative angles.
    """
    negative_lines = []
    for line in lines:
        angle = calculate_line_angle(line[0])
        if angle < 0:  # Only keep lines with negative angles
            negative_lines.append(line)
    return negative_lines

def group_lines_by_angle(lines, angle_threshold=1):  # Reduced threshold to 1 degree
    """
    Group lines into clusters based on their angles.
    """
    if lines is None:
        return {}

    # Calculate angles for all lines
    angles_with_lines = [(calculate_line_angle(line[0]), line) for line in lines]
    
    # Sort by angle
    angles_with_lines.sort(key=lambda x: x[0])
    
    # Initialize clusters
    clusters = {}
    current_cluster_angle = None
    
    for angle, line in angles_with_lines:
        if current_cluster_angle is None:
            # Start first cluster
            current_cluster_angle = angle
            clusters[angle] = [line]
        else:
            # Check if angle belongs to current cluster
            if abs(angle - current_cluster_angle) <= angle_threshold:
                clusters[current_cluster_angle].append(line)
            else:
                # Start new cluster
                current_cluster_angle = angle
                clusters[angle] = [line]
    
    # Calculate and set mean angle for each cluster
    final_clusters = {}
    for angle, cluster_lines in clusters.items():
        cluster_angles = [calculate_line_angle(line[0]) for line in cluster_lines]
        mean_angle = np.mean(cluster_angles)
        final_clusters[mean_angle] = cluster_lines
    
    return final_clusters

def filter_clusters(clusters, min_lines=5):
    """
    Filter clusters to keep only those with at least `min_lines` lines.
    
    Args:
        clusters: Dictionary of clusters (output of `group_lines_by_angle`).
        min_lines: Minimum number of lines required to keep a cluster.
    
    Returns:
        filtered_clusters: Dictionary of filtered clusters.
    """
    filtered_clusters = {}
    for angle, lines in clusters.items():
        if len(lines) >= min_lines:
            filtered_clusters[angle] = lines
    return filtered_clusters

def draw_lines(image, clusters, vanishing_point):
    """
    Draw extended lines from valid clusters on the image.
    
    Args:
        image: Input image.
        clusters: Dictionary of filtered clusters.
        vanishing_point: (x, y) coordinates of the vanishing point.
    
    Returns:
        image_with_lines: Image with extended lines drawn.
    """
    for angle, lines in clusters.items():
        # Extend lines to the vanishing point
        extended_lines = extend_lines_to_vanishing_point(lines, vanishing_point)
        
        # Draw the extended lines
        for line in extended_lines:
            x1, y1, x2, y2 = line[0]
            # Ensure the points are integers
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            # Draw the line
            cv2.line(image, pt1, pt2, (0, 0, 255), 2)
            
            # Calculate the midpoint of the line
            midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Prepare the text to display
            text = f"Angle: {angle:.2f}°"
            
            # Draw the text near the midpoint of the line
            cv2.putText(image, text, (int(midpoint[0]), int(midpoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def get_vertical_vanishing_point(lines):
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom != 0:
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                intersections.append((px, py))
    
    if intersections:
        vanishing_point = np.mean(intersections, axis=0)
        return vanishing_point
    else:
        return None

def print_lines_and_angles(lines):
    """
    Print the coordinates and angles of the detected lines.
    
    Args:
        lines: List of detected lines.
    """
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            angle = calculate_line_angle(line[0])
            print(f"Line {i + 1}: ({x1}, {y1}) -> ({x2}, {y2}), Angle: {angle:.2f}°")
    else:
        print("No lines detected.")

def extend_lines_to_vanishing_point(lines, vanishing_point):
    """
    Extend lines to the vanishing point.
    
    Args:
        lines: List of lines to extend.
        vanishing_point: (x, y) coordinates of the vanishing point.
    
    Returns:
        extended_lines: List of extended lines.
    """
    extended_lines = []
    vx, vy = vanishing_point

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate the slope of the line
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
        else:
            slope = float('inf')  # Vertical line

        # Extend the line to the vanishing point
        if slope != float('inf'):
            # Calculate the new start and end points
            new_x1 = min(x1, x2, vx)  # Start at the leftmost point
            new_y1 = int(y1 + slope * (new_x1 - x1))
            new_x2 = max(x1, x2, vx)  # End at the rightmost point
            new_y2 = int(y1 + slope * (new_x2 - x1))
        else:
            # Vertical line: extend to the vanishing point's y-coordinate
            new_x1 = x1
            new_y1 = min(y1, y2, vy)
            new_x2 = x1
            new_y2 = max(y1, y2, vy)

        extended_lines.append([[int(new_x1), int(new_y1), int(new_x2), int(new_y2)]])

    return extended_lines

def find_lines_intersection(lines):
    """
    Find the intersection point of multiple lines using least squares method.
    """
    A = []
    b = []
    
    for line in lines:
        x1, y1 = line[0], line[1]
        x2, y2 = line[2], line[3]
        
        if x2 != x1:  # non-vertical line
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            A.append([-m, 1])
            b.append(c)
    
    if len(A) >= 2:
        A = np.array(A)
        b = np.array(b)
        try:
            intersection = np.linalg.lstsq(A, b, rcond=None)[0]
            return (int(intersection[0]), int(intersection[1]))
        except:
            return None
    return None

def visualize_vanishing_point(image, clusters, vanishing_point, output_path='vanishing_point_visualization.jpg', show_window=True, detections=None):
    height, width = image.shape[:2]
    scale_factor = 0.3
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    canvas_height = new_height * 2
    canvas_width = new_width * 3
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    resized_image = cv2.resize(image, (new_width, new_height))
    y_offset = canvas_height - new_height
    canvas[y_offset:canvas_height, 0:new_width] = resized_image

    representative_lines = []
    for angle, lines in clusters.items():
        mean_x = np.mean([line[0][0] for line in lines])
        representative_line = min(lines, key=lambda line: abs(line[0][0] - mean_x))
        x1, y1, x2, y2 = representative_line[0]
        
        scaled_x1 = int(x1 * scale_factor)
        scaled_y1 = int(y1 * scale_factor) + y_offset
        scaled_x2 = int(x2 * scale_factor)
        scaled_y2 = int(y2 * scale_factor) + y_offset
        
        representative_lines.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])

    intersection = find_lines_intersection(representative_lines)
    if intersection:
        vx, vy = intersection
        print(f"\nVanishing point: ({vx}, {vy})")
        
        for line in representative_lines:
            x1, y1, x2, y2 = line
            cv2.line(canvas, (x1, y1), (vx, vy), (0, 0, 255), 2)
        
        if detections:
            defenders = [d for d in detections if d.get('team') == 'team1']
            attackers = [d for d in detections if d.get('team') == 'team2']
            
            if defenders:
                # First identify the goalkeeper (leftmost player)
                goalkeeper = min(defenders, key=lambda d: d['bbox'][0])
                # Remove goalkeeper from calculations
                defenders = [d for d in defenders if d != goalkeeper]
                
                # Calculate angles for remaining defenders
                defender_angles = []
                for defender in defenders:
                    x1, y1, x2, y2 = defender['bbox']
                    foot_x = int(x1)
                    foot_y = int(y2)
                    
                    # Calculate angle between vertical line and line to player
                    angle = np.arctan2(foot_x - vx, vy - foot_y)
                    if angle < 0:
                        angle += 2 * np.pi
                    
                    # Calculate distance from vanishing point
                    distance = np.sqrt((foot_x - vx)**2 + (foot_y - vy)**2)
                    
                    defender_angles.append((defender, angle, distance))
                    print(f"Position: ({foot_x}, {foot_y}), Angle: {np.degrees(angle):.2f}°, Distance: {distance:.2f}")
                
                # Sort defenders by angle and distance weight
                max_distance = max(d[2] for d in defender_angles)
                weighted_positions = []
                for defender, angle, distance in defender_angles:
                    normalized_distance = distance / max_distance
                    weight_score = np.degrees(angle) * (1 - 0.3 * normalized_distance)
                    weighted_positions.append((defender, weight_score))
                
                sorted_defenders = sorted(weighted_positions, key=lambda x: x[1], reverse=True)
                last_defender = sorted_defenders[0][0] if sorted_defenders else None

                # Draw all player lines and analyze offside
                for detection in detections:
                    if detection.get('team') == 'referee':
                        continue
                        
                    x1, y1, x2, y2 = detection['bbox']
                    foot_x = int(x1)
                    foot_y = int(y2)
                    
                    scaled_foot_x = int(foot_x * scale_factor)
                    scaled_foot_y = int(foot_y * scale_factor) + y_offset
                    
                    if last_defender and detection == last_defender:
                        color = (0, 255, 255)  # Yellow for last defender
                        thickness = 3
                    else:
                        color = (255, 0, 0) if detection.get('team') == 'team1' else (0, 255, 0)
                        thickness = 1
                    
                    # Draw line to vanishing point
                    cv2.line(canvas, (scaled_foot_x, scaled_foot_y), (vx, vy), color, thickness)
                    cv2.circle(canvas, (scaled_foot_x, scaled_foot_y), 2, color, -1)
                    
                    # Add offside analysis text for attacking players
                    if detection.get('team') == 'team2' and last_defender:
                        # Calculate angles for offside check
                        attacker_angle = np.arctan2(foot_x - vx, vy - foot_y)
                        if attacker_angle < 0:
                            attacker_angle += 2 * np.pi
                        
                        last_def_x, last_def_y = last_defender['bbox'][0], last_defender['bbox'][3]
                        defender_angle = np.arctan2(last_def_x - vx, vy - last_def_y)
                        if defender_angle < 0:
                            defender_angle += 2 * np.pi
                            
                        # Compare angles for offside 
                        is_offside = attacker_angle > defender_angle  
                        
                        # Add text above the player
                        text = "OFFSIDE" if is_offside else "ONSIDE"
                        text_color = (0, 0, 255) if is_offside else (0, 255, 0)
                        text_pos = (scaled_foot_x - 30, scaled_foot_y - 10)
                        cv2.putText(canvas, text, text_pos, 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        cv2.circle(canvas, (vx, vy), 5, (0, 0, 255), -1)
    
    cv2.putText(canvas, "Offside Analysis", 
                (canvas_width//4, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 0, 0), 2)

    cv2.imwrite(output_path, canvas)
    
    if show_window:
        cv2.imshow('Vanishing Point Visualization', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return canvas
def draw_boxes_on_visualization(image, detections, scale_factor=0.3, debug=False):
    """Draw bounding boxes on the visualization canvas with proper scaling"""
    result = image.copy()
    canvas_height, canvas_width = result.shape[:2]
    
    # Calculate the region where the scaled image is located
    new_width = int(canvas_width / 3)  # The image takes up 1/3 of the canvas width
    new_height = int(canvas_height / 2)  # The image takes up 1/2 of the canvas height
    
    colors = {
        'team1': (255, 0, 0),   # Blue for team 1
        'team2': (0, 0, 255),   # Red for team 2
        'referee': (0, 255, 0),  # Green for referee
        'ball': (0, 255, 255),   # Yellow for ball
        'unknown': (128, 128, 128)  # Gray for unknown
    }
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        
        # Scale coordinates to match the scaled image region
        x1 = int((x1 * new_width) / 1920) 
        x2 = int((x2 * new_width) / 1920)
        y1 = int((y1 * new_height) / 1080)  
        y2 = int((y2 * new_height) / 1080)
        
        # Adjust y coordinates to bottom half of canvas
        y1 = canvas_height - new_height + y1
        y2 = canvas_height - new_height + y2
        
        team = detection.get('team', 'unknown')
        confidence = detection.get('confidence', 0)
        
        color = colors.get(team, colors['unknown'])
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        if debug:
            label = f"{team} ({confidence:.2f})"
            cv2.putText(result, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result

def detect_horizontal_lines(image):
    """Detect horizontal field lines"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color (field)
    lower_green = np.array([36, 0, 0])
    upper_green = np.array([86, 255, 255])
    
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Transform
    # Adjust parameters to detect more horizontal lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=50, maxLineGap=10)
    
    return lines

def filter_horizontal_angles(lines, angle_threshold=30):
    """Filter lines to keep only those close to horizontal"""
    horizontal_lines = []
    if lines is None:
        return horizontal_lines
        
    for line in lines:
        angle = abs(calculate_line_angle(line[0]))
        # Keep lines that are close to horizontal (0 or 180 degrees)
        if angle < angle_threshold or angle > (180 - angle_threshold):
            horizontal_lines.append(line)
    return horizontal_lines

def get_horizontal_vanishing_point(image):
    """Get the horizontal vanishing point"""
    # Detect horizontal lines
    lines = detect_horizontal_lines(image)
    if lines is None:
        return None
        
    # Filter for horizontal lines
    horizontal_lines = filter_horizontal_angles(lines)
    if len(horizontal_lines) < 2:
        return None
    
    # Find intersections of horizontal lines
    intersections = []
    for i, line1 in enumerate(horizontal_lines):
        for line2 in horizontal_lines[i+1:]:
            x1, y1, x2, y2 = line1[0]
            x3, y3, x4, y4 = line2[0]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom != 0:
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                # Only keep intersections that are within reasonable bounds
                if -1000 < px < image.shape[1] + 1000 and -1000 < py < image.shape[0] + 1000:
                    intersections.append((px, py))
    
    if intersections:
        # Remove outliers before taking mean
        intersections = np.array(intersections)
        mean = np.mean(intersections, axis=0)
        std = np.std(intersections, axis=0)
        filtered_intersections = intersections[np.all(np.abs(intersections - mean) < 2 * std, axis=1)]
        
        if len(filtered_intersections) > 0:
            vanishing_point = np.mean(filtered_intersections, axis=0)
            return vanishing_point
    
    return None

def line_intersection(line1, line2):
    """Find intersection point of two lines"""
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    
    # Calculate denominators
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:  # Lines are parallel
        return None
        
    # Calculate intersection point
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
    
    return (int(px), int(py))

def get_ground_plane(image, vertical_vanishing_point, horizontal_vanishing_point, side):
    """Calculate ground plane parameters"""
    x1, y1 = vertical_vanishing_point
    x2, y2 = horizontal_vanishing_point
    z1 = z2 = 0
    
    # Get a third point from field lines intersection
    lines = detect_field_lines(image)
    vertical_lines = [l for l in lines if abs(np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]))) > 45]
    horizontal_lines = [l for l in lines if abs(np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]))) <= 45]
    
    if len(vertical_lines) > 0 and len(horizontal_lines) > 0:
        x3, y3 = line_intersection(vertical_lines[0], horizontal_lines[0])
    else:
        x3, y3 = (image.shape[1]/2, image.shape[0])
    z3 = 0
    
    # Calculate plane parameters
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (-a * x1 - b * y1 - c * z1)
    
    return [a, b, c, d]