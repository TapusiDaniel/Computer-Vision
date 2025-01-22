import cv2

def visualize_results(image, pose_estimations, offside_decisions, vanishing_point):
    for pose in pose_estimations:
        x, y = pose['position']
        if pose['team'] == 'attacking':
            color = (0, 255, 0)  # Green for attacking team
        else:
            color = (0, 0, 255)  # Red for defending team
        
        cv2.circle(image, (int(x), int(y)), 5, color, -1)
    
    if vanishing_point:
        cv2.circle(image, (int(vanishing_point[0]), int(vanishing_point[1])), 10, (255, 0, 0), -1)
    
    for decision in offside_decisions:
        if decision['offside']:
            x, y = decision['position']
            cv2.putText(image, 'Offside', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)