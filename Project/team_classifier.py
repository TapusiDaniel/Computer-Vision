import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class TeamClassifier:
    def __init__(self):
        self.debug = False

    def extract_jersey_color(self, image, bbox):
        """Extract the dominant color from player's jersey area."""
        x1, y1, x2, y2 = bbox
        
        # Focus on the upper body area (jersey)
        height = y2 - y1
        jersey_top = y1 + int(height * 0.2)
        jersey_bottom = y1 + int(height * 0.5)
        jersey_roi = image[jersey_top:jersey_bottom, x1:x2]
        
        if jersey_roi.size == 0:
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(jersey_roi, cv2.COLOR_BGR2HSV)
        
        # Reshape for clustering
        pixels = hsv.reshape(-1, 3)
        
        if pixels.shape[0] < 1:
            return None

        # Use k-means to find the dominant color
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]

        return dominant_color

    def get_color_name(self, hsv_color):
        """Convert HSV color to basic color name."""
        h, s, v = hsv_color
        
        # More comprehensive referee (dark colors) detection
        if v < 80 or (v < 100 and s < 80):  # Expanded dark color detection
            return "black"
        elif v > 200 and s < 50:
            return "white"
        
        # Normalize hue to 0-360 range
        h = h * 2  # OpenCV uses 0-180 for hue
        
        if s < 40:
            return "gray"
        
        if h < 30 or h > 330:
            return "red"
        elif 30 <= h < 90:
            if v > 150 and s > 100:
                return "yellow"
            return "green"
        elif 90 <= h < 150:
            return "light blue"
        elif 150 <= h < 210:
            if v < 120: 
                return "black"
            return "blue"
        elif 210 <= h < 270:
            return "purple"
        elif 270 <= h < 330:
            return "pink"
        
        return "unknown"

    def is_likely_referee_position(self, bbox, image_width, image_height):
        """Check if the position is typical for a referee."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if position is near sidelines
        near_sideline = center_x < image_width * 0.1 or center_x > image_width * 0.9
        
        return near_sideline

    def cluster_teams(self, image, detections):
        """Group players by jersey color and assign teams."""
        image_height, image_width = image.shape[:2]
        color_groups = {}
        dark_players = []
        
        # First pass: collect all players and their colors
        for det in detections:
            if det.get('class_id') == 0:  # person class
                color = self.extract_jersey_color(image, det['bbox'])
                if color is not None:
                    color_name = self.get_color_name(color)
                    if color_name == "black":
                        dark_players.append((det, color))
                    else:
                        if color_name not in color_groups:
                            color_groups[color_name] = []
                        color_groups[color_name].append(det)

        # Print initial color groups
        print("\nColor Groups Found:")
        for color, players in color_groups.items():
            print(f"{color}: {len(players)} players")
        print(f"dark/black: {len(dark_players)} players")

        # Handle dark-colored players (potential referees)
        referees = []
        for det, color in dark_players:
            # If we already found two referees, treat remaining dark players normally
            if len(referees) < 2:
                referees.append(det)
            else:
                color_name = "blue"  # Fallback color for non-referee dark players
                if color_name not in color_groups:
                    color_groups[color_name] = []
                color_groups[color_name].append(det)

        # Sort remaining groups by size
        sorted_groups = sorted(color_groups.items(), key=lambda x: len(x[1]), reverse=True)

        # Get the two largest non-referee groups for teams
        team_groups = sorted_groups[:2] if len(sorted_groups) >= 2 else sorted_groups

        print("\nTeam Assignments:")
        if team_groups:
            print(f"Team 1 (main team): {team_groups[0][0]}")
            if len(team_groups) > 1:
                print(f"Team 2: {team_groups[1][0]}")
        print(f"Referees found: {len(referees)}")

        # Assign referees
        for ref in referees:
            ref['team'] = 'referee'
            print(f"Assigned referee")

        # Assign main teams
        if team_groups:
            # Team 1 (largest group)
            for player in team_groups[0][1]:
                player['team'] = 'team1'
                print(f"Assigned to team1 (color: {team_groups[0][0]})")

            # Team 2 (second largest group)
            if len(team_groups) > 1:
                for player in team_groups[1][1]:
                    player['team'] = 'team2'
                    print(f"Assigned to team2 (color: {team_groups[1][0]})")

        # Handle any remaining players (like goalkeeper)
        for color, players in color_groups.items():
            if color not in [g[0] for g in team_groups]:
                # Assign to team with most players
                team1_count = len(team_groups[0][1]) if team_groups else 0
                team2_count = len(team_groups[1][1]) if len(team_groups) > 1 else 0
                team = 'team1' if team1_count > team2_count else 'team2'
                for player in players:
                    player['team'] = team
                    print(f"Assigned {color} player to {team} (based on team size)")

        return detections