# GhostFace-AI-Recognizing-Faces-Without-Prior-Knowledge
import cv2
import numpy as np
import os
import pickle

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images - lower means more similar"""
    # Ensure images are the same size
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    
    # Convert to grayscale for simpler comparison
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate MSE
    err = np.sum((img1_gray.astype("float") - img2_gray.astype("float")) ** 2)
    err /= float(img1_gray.shape[0] * img1_gray.shape[1])
    
    return err

def main():
    # Create storage directory if it doesn't exist
    storage_dir = "stored_images"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    
    # Load existing data if available
    data_file = "image_data.pkl"
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            stored_images = pickle.load(f)
        print(f"Loaded {len(stored_images)} existing images")
    else:
        stored_images = []
    
    # Step 1: Store multiple images with metadata
    print("=== STORE MULTIPLE IMAGES ===")
    while True:
        image_path = input("Enter image path (or 'done' to finish): ")
        if image_path.lower() == 'done':
            break
        
        if not os.path.exists(image_path):
            print("File not found. Please try again.")
            continue
        
        # Get additional data from user
        name = input("Enter a name for this image: ")
        description = input("Enter a description: ")
        
        # Load and store the image
        img = cv2.imread(image_path)
        if img is None:
            print("Could not read image. Please try a different file.")
            continue
        
        # Save image to storage directory
        image_id = len(stored_images)
        filename = f"{storage_dir}/image_{image_id}.jpg"
        cv2.imwrite(filename, img)
        
        # Store metadata
        stored_images.append({
            'id': image_id,
            'filename': filename,
            'name': name,
            'description': description
        })
        
        print(f"Image stored successfully as ID {image_id}!")
    
    # Save the updated data
    with open(data_file, 'wb') as f:
        pickle.dump(stored_images, f)
    
    # Step 2: Match a new image against stored images
    print("\n=== MATCH NEW IMAGE ===")
    new_image_path = input("Enter path to the image you want to match: ")
    
    if not os.path.exists(new_image_path):
        print("File not found.")
        return
    
    new_img = cv2.imread(new_image_path)
    if new_img is None:
        print("Could not read image.")
        return
    
    # Compare with stored images
    best_match = None
    best_score = float('inf')  # Lower MSE is better
    threshold = 1000  # Threshold for considering a match (adjust as needed)
    
    for entry in stored_images:
        stored_img = cv2.imread(entry['filename'])
        score = calculate_mse(new_img, stored_img)
        
        if score < best_score:
            best_score = score
            best_match = entry
    
    # Check if we found a good match
    if best_match and best_score < threshold:
        print(f"Found a match! Image ID: {best_match['id']}")
        print(f"Name: {best_match['name']}")
        print(f"Description: {best_match['description']}")
        print(f"Similarity score (lower is better): {best_score:.2f}")
    else:
        print("No match found. Let's store this new image.")
        name = input("Enter a name for this image: ")
        description = input("Enter a description: ")
        
        # Save the new image
        image_id = len(stored_images)
        filename = f"{storage_dir}/image_{image_id}.jpg"
        cv2.imwrite(filename, new_img)
        
        # Store metadata
        stored_images.append({
            'id': image_id,
            'filename': filename,
            'name': name,
            'description': description
        })
        
        # Save the updated data
        with open(data_file, 'wb') as f:
            pickle.dump(stored_images, f)
        
        print(f"New image stored successfully as ID {image_id}!")

if __name__ == "__main__":
    main()
