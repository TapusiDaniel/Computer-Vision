import os
import cv2
import numpy as np
from pathlib import Path

class EigenfacesRecognition:
    def __init__(self, n_components=None, threshold=None):
        self.n_components = n_components
        self.threshold = threshold
        self.mean_face = None
        self.eigenfaces = None
        self.weights = None
        self.training_faces = None
        self.training_ids = None

    def train(self, train_path):
        """Train the eigenfaces recognizer following Turk and Pentland's method"""

        training_images = []
        self.training_ids = []        
        train_files = list(Path(train_path).glob('*.jpg'))
        
        # Read all training images
        for img_path in train_files:
            person_id = int(img_path.name.split('_')[0])
            img = cv2.imread(str(img_path))
            training_images.append(img.flatten())
            self.training_ids.append(person_id)
            
        # Convert to numpy array
        self.training_faces = np.array(training_images)
        self.training_ids = np.array(self.training_ids)
        
        # Calculate mean face 
        self.mean_face = np.mean(self.training_faces, axis=0)
        
        # Center the faces (mean = 0)
        centered_faces = self.training_faces - self.mean_face
        
        # Calculate covariance matrix using the trick from the paper
        # L = A^T * A where A is the centered faces matrix
        L = np.dot(centered_faces, centered_faces.T)
        
        # Calculate eigenvectors of L
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort eigenvectors by eigenvalues in descending order
        # The eigenvalues with the highest values correspond to the maximum variation in the data
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate eigenfaces (eigenvectors of the covariance matrix)
        self.eigenfaces = np.dot(centered_faces.T, eigenvectors)
        
        # Normalize eigenfaces, to have the norm of each eigenface equal to 1
        for i in range(self.eigenfaces.shape[1]):
            self.eigenfaces[:, i] = self.eigenfaces[:, i] / np.linalg.norm(self.eigenfaces[:, i])
            
        # Keep only the selected number of eigenfaces
        self.eigenfaces = self.eigenfaces[:, :self.n_components]
        
        # Calculate weights for training faces (project faces into the subspace of eigenfaces)
        self.weights = np.dot(centered_faces, self.eigenfaces)

    def predict(self, img_path):
        """Recognize a face using eigenfaces method"""
        
        # Read and preprocess test image
        img = cv2.imread(img_path)
        test_face = img.flatten()
        
        # Center the faces (mean = 0)
        centered_test = test_face - self.mean_face
        
        # Project test face into face space
        test_weights = np.dot(centered_test, self.eigenfaces)
        
        # Calculate distances to all training faces in face space
        # We calculate the euclidean distance between the weights of the test face and the weights of the training faces
        distances = np.linalg.norm(self.weights - test_weights, axis=1)
        
        # Find the closest training face to the test face
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # Calculate distance from face space (reconstruction error)
        # The distance between the test face and its reconstruction in the face space
        # A big error means that the test face is not well represented by the eigenfaces
        reconstruction = self.mean_face + (np.dot(self.eigenfaces, test_weights))
        face_space_distance = np.linalg.norm(centered_test - (np.dot(self.eigenfaces, test_weights)))
            
        return self.training_ids[min_distance_idx], min_distance, face_space_distance

def evaluate_recognizer(train_path, test_path):
    """Evaluate the eigenfaces recognizer"""
    
    # Initialize and train recognizer
    recognizer = EigenfacesRecognition(n_components=30)
    recognizer.train(train_path)
    
    # Test the recognizer
    correct = 0
    total = 0
    test_files = list(Path(test_path).glob('*.jpg'))
    results = []
    
    for img_path in test_files:
        true_person = int(img_path.name.split('_')[0])
        predicted_person, distance, face_space_distance = recognizer.predict(str(img_path))
        
        results.append({
            'true_person': true_person,
            'predicted_person': predicted_person,
            'distance': distance,
            'face_space_distance': face_space_distance
        })
        
        if true_person == predicted_person:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy, results, recognizer.n_components

if __name__ == "__main__":
    # Set paths
    train_path = "Faces_training_and_testing_images/Train"
    test_path = "Faces_training_and_testing_images/Test"
    
    # Train and evaluate
    accuracy, results, n_components = evaluate_recognizer(train_path, test_path)
    
    # Print results
    print(f"Number of components used: {n_components}")
    print(f"Classification accuracy: {accuracy * 100:.2f}%")
    
    # Print detailed results
    # print("\nDetailed results:")
    # for result in results:
    #     print(f"True person: {result['true_person']}, "
    #           f"Predicted: {result['predicted_person']}, "
    #           f"Distance: {result['distance']:.2f}, "
    #           f"Face space distance: {result['face_space_distance']:.2f}")
