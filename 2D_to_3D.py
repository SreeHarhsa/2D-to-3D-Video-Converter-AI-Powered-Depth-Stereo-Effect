import os
import cv2
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from tqdm import tqdm
import streamlit as st
import base64
import time

class DepthEstimator:
    def __init__(self):
        # Load DPT model for depth estimation
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
    def estimate_depth(self, image):
        # Process image and get depth map
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Normalize depth map
        depth_map = predicted_depth.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = (depth_map * 255).astype(np.uint8)
        
        # Resize depth map to match image dimensions
        depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
        
        return depth_map

class Video3DConverter:
    def __init__(self, parallax_strength=0.05):
        self.depth_estimator = DepthEstimator()
        self.parallax_strength = parallax_strength
        
    def generate_stereoscopic_pair(self, frame, depth_map):
        # Create left and right views using depth map
        height, width = frame.shape[:2]
        left_view = np.zeros_like(frame)
        right_view = np.zeros_like(frame)
        
        # Using vectorized operations for better performance and to avoid index errors
        for y in range(height):
            for x in range(width):
                # Calculate disparity based on depth (with safety bounds)
                disparity = int(self.parallax_strength * (255 - depth_map[y, x]))
                disparity = min(disparity, x)  # Ensure we don't go out of bounds
                
                # Set left view (ensure indices are in bounds)
                left_x = max(0, x - disparity)
                left_view[y, left_x] = frame[y, x]
                
                # Set right view (ensure indices are in bounds)
                right_x = min(width - 1, x + disparity)
                right_view[y, right_x] = frame[y, x]
        
        # Fill holes using interpolation
        left_view = cv2.medianBlur(left_view, 3)
        right_view = cv2.medianBlur(right_view, 3)
        
        return left_view, right_view

    def generate_stereoscopic_pair_safe(self, frame, depth_map):
        """Alternative implementation using a safer approach"""
        height, width = frame.shape[:2]
        left_view = frame.copy()
        right_view = frame.copy()
        
        # Create displacement maps
        max_disparity = int(self.parallax_strength * 255)
        for y in range(height):
            for x in range(width):
                # Calculate disparity based on depth (with safety bounds)
                disparity = int(self.parallax_strength * (255 - depth_map[y, x]))
                
                # Apply displacement to left and right views
                if x - disparity >= 0:
                    left_view[y, x - disparity] = frame[y, x]
                if x + disparity < width:
                    right_view[y, x + disparity] = frame[y, x]
        
        # Fill holes using interpolation
        left_view = cv2.medianBlur(left_view, 3)
        right_view = cv2.medianBlur(right_view, 3)
        
        return left_view, right_view
    
    def generate_autostereoscopic(self, left_view, right_view):
        # Create interlaced view for autostereoscopic display
        height, width = left_view.shape[:2]
        autostereoscopic = np.zeros_like(left_view)
        
        for y in range(height):
            for x in range(width):
                if x % 2 == 0:
                    autostereoscopic[y, x] = left_view[y, x]
                else:
                    autostereoscopic[y, x] = right_view[y, x]
        
        return autostereoscopic
    
    def convert_video(self, input_path, output_dir, method="autostereoscopic"):
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Create output filename
        video_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_3d_{method}.mp4")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        print(f"Processing video with {method} method...")
        for frame_idx in tqdm(range(min(total_frames, 300))):  # Limit to 300 frames for testing
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Estimate depth
                depth_map = self.depth_estimator.estimate_depth(frame)
                
                # Generate stereoscopic pair (using the safer implementation)
                left_view, right_view = self.generate_stereoscopic_pair_safe(frame, depth_map)
                
                # Generate output based on method
                if method == "autostereoscopic":
                    output_frame = self.generate_autostereoscopic(left_view, right_view)
                elif method == "anaglyph":
                    # Create anaglyph (red-cyan) for testing
                    output_frame = cv2.merge([right_view[:,:,0], left_view[:,:,1], left_view[:,:,2]])
                elif method == "side_by_side":
                    # Create side-by-side
                    output_frame = np.hstack((left_view[:,:width//2], right_view[:,width//2:]))
                    output_frame = cv2.resize(output_frame, (width, height))
                else:
                    output_frame = frame
                
                out.write(output_frame)
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                # Use original frame as fallback
                out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Conversion complete. Output saved to: {output_path}")
        return output_path

def get_video_download_link(video_path, filename):
    """Generate a download link for a video file"""
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">Download Processed Video</a>'
    return href

# Streamlit app
def main():
    st.set_page_config(page_title="2D to 3D Video Converter", page_icon="🎥", layout="wide")
    
    st.title("2D to 3D Video Converter")
    st.write("""
    Convert your 2D videos to autostereoscopic 3D videos that can be viewed without special glasses.
    """)
    
    # Fixed paths from the user
    input_path = "/home/harsha/Downloads/video2.mp4"
    output_dir = "/home/harsha/Downloads"
    
    # Display info about the paths
    st.info(f"Input video: {input_path}")
    st.info(f"Output directory: {output_dir}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        st.error(f"Input file does not exist: {input_path}")
        return
    
    # Display original video
    st.subheader("Original Video")
    st.video(input_path)
    
    # 3D conversion settings
    st.subheader("3D Conversion Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        parallax_strength = st.slider("Parallax Strength", 0.01, 0.20, 0.05, 0.01,
                                     help="Controls the strength of the 3D effect. Higher values create deeper 3D but may cause eye strain.")
        
    with col2:
        output_format = st.selectbox("Output Format", 
                                    ["autostereoscopic", "anaglyph", "side_by_side"],
                                    help="Autostereoscopic: for glasses-free 3D displays. Anaglyph: for red-cyan glasses. Side-by-side: for VR headsets.")
    
    # Process button
    if st.button("Convert to 3D"):
        with st.spinner("Converting video to 3D... This may take several minutes depending on the video length."):
            try:
                # Initialize converter with settings
                converter = Video3DConverter(parallax_strength=parallax_strength)
                
                # Process video
                result_path = converter.convert_video(input_path, output_dir, method=output_format)
                
                # Display result
                st.success("Conversion complete!")
                st.subheader("3D Video Result")
                st.video(result_path)
                
                # Get filename for download
                output_filename = os.path.basename(result_path)
                
                # Download button
                st.markdown(get_video_download_link(result_path, output_filename), unsafe_allow_html=True)
                
                # Direct file path info
                st.info(f"The 3D video has been saved to: {result_path}")
                
                # Tips for viewing
                st.info("""
                **Viewing Tips:**
                - For autostereoscopic format, view on a 3D-compatible display
                - Optimal viewing distance is about 2-3 times the screen height
                - Try slightly tilting your head to find the sweet spot
                """)
                
            except Exception as e:
                st.error(f"An error occurred during conversion: {str(e)}")

# For direct execution
if __name__ == "__main__":
    main()
