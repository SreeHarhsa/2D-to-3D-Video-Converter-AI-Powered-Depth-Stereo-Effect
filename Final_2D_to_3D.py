import os
import cv2
import numpy as np
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
import streamlit as st
import base64
import time
import gc
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import tempfile
from PIL import Image
import torch.nn.functional as F

class FastDepthEstimator:
    def __init__(self, model_name="Intel/dpt-hybrid-midas", device=None):
        # Use smaller hybrid model for faster inference
        self.model_name = model_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Initializing depth model {model_name} on {self.device}")
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        
        # Move model to appropriate device and optimize
        self.model.to(self.device)
        if self.device.type == 'cuda':
            self.model = self.model.half()  # Use half precision for faster inference
            
        # Model optimization
        self.model.eval()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # For batch processing
        self.batch_size = 2 if self.device.type == 'cuda' else 1
        
        # Cached depth maps for keyframes to speed up processing
        self.depth_cache = {}
        self.keyframe_interval = 5  # Process every 5th frame in full detail
        
    def estimate_depth(self, image, frame_idx=None):
        # Check cache first if frame_idx is provided
        if frame_idx is not None and frame_idx in self.depth_cache:
            return self.depth_cache[frame_idx]
            
        # Determine if this is a keyframe for detailed processing
        is_keyframe = frame_idx is None or frame_idx % self.keyframe_interval == 0
        
        # Resize for faster processing if not a keyframe
        h, w = image.shape[:2]
        if not is_keyframe:
            # Use lower resolution for non-keyframes
            scale = 0.5
            small_img = cv2.resize(image, (int(w * scale), int(h * scale)))
            process_img = small_img
        else:
            process_img = image
            
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):
            # Process image and get depth map
            inputs = self.processor(images=process_img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Normalize depth map
        depth_map = predicted_depth.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = (depth_map * 255).astype(np.uint8)
        
        # For non-keyframes, resize back to original size
        if not is_keyframe:
            depth_map = cv2.resize(depth_map, (w, h))
            
            # If there's a previous keyframe, blend for temporal consistency
            prev_keyframe = (frame_idx // self.keyframe_interval) * self.keyframe_interval
            if prev_keyframe in self.depth_cache:
                prev_depth = self.depth_cache[prev_keyframe]
                # Blend with previous keyframe for temporal stability
                blend_factor = (frame_idx - prev_keyframe) / self.keyframe_interval
                depth_map = cv2.addWeighted(prev_depth, 1-blend_factor, depth_map, blend_factor, 0)
        
        # Cache keyframes for future reference
        if is_keyframe and frame_idx is not None:
            self.depth_cache[frame_idx] = depth_map.copy()
            
            # Clear old cache entries to avoid memory issues
            if len(self.depth_cache) > 30:  # Keep only last 30 keyframes
                oldest_key = min(self.depth_cache.keys())
                del self.depth_cache[oldest_key]
                
        return depth_map
        
    def batch_estimate_depth(self, images, frame_indices=None):
        """Process multiple frames in batch for efficiency"""
        if not images:
            return []
            
        if frame_indices is None:
            frame_indices = [None] * len(images)
            
        # Split into batches
        batches = [(images[i:i+self.batch_size], frame_indices[i:i+self.batch_size]) 
                  for i in range(0, len(images), self.batch_size)]
        
        all_depths = []
        for batch_images, batch_indices in batches:
            # Process uncached frames
            uncached_images = []
            uncached_indices = []
            cached_depths = []
            
            for img, idx in zip(batch_images, batch_indices):
                if idx is not None and idx in self.depth_cache:
                    cached_depths.append(self.depth_cache[idx])
                else:
                    uncached_images.append(img)
                    uncached_indices.append(idx)
            
            # Process uncached images
            if uncached_images:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):
                    # Process images in batch
                    inputs = self.processor(images=uncached_images, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    predicted_depths = outputs.predicted_depth
                
                # Process each depth map
                for i, (depth_tensor, idx) in enumerate(zip(predicted_depths, uncached_indices)):
                    depth_map = depth_tensor.cpu().numpy()
                    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                    depth_map = (depth_map * 255).astype(np.uint8)
                    
                    # Resize to match original if needed
                    if depth_map.shape[:2] != uncached_images[i].shape[:2]:
                        depth_map = cv2.resize(depth_map, 
                                             (uncached_images[i].shape[1], uncached_images[i].shape[0]))
                    
                    # Cache keyframes
                    if idx is not None and idx % self.keyframe_interval == 0:
                        self.depth_cache[idx] = depth_map
                        
                    all_depths.append(depth_map)
            
            # Add cached depths
            all_depths.extend(cached_depths)
        
        return all_depths
            
    def release_resources(self):
        """Clean up resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        self.depth_cache = {}

class OptimizedVideo3DConverter:
    def __init__(self, parallax_strength=0.05, edge_enhancement=0.0, 
                model_name="Intel/dpt-hybrid-midas", preview_mode=False):
        # Use a more optimized model by default
        self.depth_estimator = FastDepthEstimator(model_name=model_name)
        self.parallax_strength = parallax_strength
        self.edge_enhancement = edge_enhancement
        self.preview_mode = preview_mode
        self.last_depth_map = None
        
    def optimize_frame(self, frame, resolution_scale=1.0):
        """Pre-process frame for faster depth estimation"""
        if resolution_scale < 1.0:
            h, w = frame.shape[:2]
            return cv2.resize(frame, (int(w * resolution_scale), int(h * resolution_scale)))
        return frame
        
    def vectorized_stereoscopic_pair(self, frame, depth_map):
        """Highly optimized stereoscopic pair generation"""
        height, width = frame.shape[:2]
    
        # Ensure depth map has the same dimensions as the frame
        if depth_map.shape[:2] != frame.shape[:2]:
            depth_map = cv2.resize(depth_map, (width, height))
    
        # Create displacement map - use numpy vectorized operations
        max_disparity = int(self.parallax_strength * 255)
        disparity_map = max_disparity * (1.0 - depth_map/255.0)
    
        # Create coordinate maps for remapping
        y_coords = np.arange(height).reshape(-1, 1).repeat(width, axis=1).astype(np.float32)
    
        # Create x-coordinate maps directly
        left_x = np.zeros((height, width), dtype=np.float32)
        right_x = np.zeros((height, width), dtype=np.float32)
    
        # Fill in coordinate maps properly ensuring shapes match
        for y in range(height):
            left_x[y] = np.arange(width) - disparity_map[y]
            right_x[y] = np.arange(width) + disparity_map[y]
    
        # Clip coordinates to valid range
        left_x = np.clip(left_x, 0, width-1).astype(np.float32)
        right_x = np.clip(right_x, 0, width-1).astype(np.float32)
    
        # Remapping - faster than loop
        left_view = cv2.remap(frame, left_x, y_coords, cv2.INTER_LINEAR)
        right_view = cv2.remap(frame, right_x, y_coords, cv2.INTER_LINEAR)
    
        return left_view, right_view
    
    def fast_autostereoscopic(self, left_view, right_view):
        """Extremely fast implementation of autostereoscopic generation"""
        # Use array indexing for maximum speed
        height, width = left_view.shape[:2]
        result = np.zeros_like(left_view)
        
        # Alternate columns
        result[:, 0::2] = left_view[:, 0::2]
        result[:, 1::2] = right_view[:, 1::2]
        
        return result
    
    def precompute_frame_indices(self, total_frames, samples=10):
        """Determine which frames to process in full vs interpolate"""
        keyframes = np.linspace(0, total_frames-1, samples).astype(int)
        keyframe_dict = {frame: True for frame in keyframes}
        return keyframe_dict
    
    def convert_video(self, input_path, output_dir, method="autostereoscopic", 
                     resolution_scale=0.5, skip_frames=1, 
                     start_frame=0, end_frame=None, max_preview_frames=None):
        """Convert video with multiple optimizations for speed"""
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Apply resolution scaling for output
        width = int(orig_width * resolution_scale)
        height = int(orig_height * resolution_scale)
        
        # Adjust end frame if specified
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
            
        # For preview mode, process fewer frames
        if self.preview_mode and max_preview_frames:
            # Process evenly distributed frames 
            preview_step = max(1, (end_frame - start_frame) // max_preview_frames)
            skip_frames = max(skip_frames, preview_step)
            
        # Number of frames to process (accounting for skipping)
        frames_to_process = (end_frame - start_frame + skip_frames - 1) // skip_frames
        
        print(f"Video info: {orig_width}x{orig_height} → {width}x{height}, {fps} fps")
        print(f"Processing {frames_to_process} out of {end_frame-start_frame} frames (skipping {skip_frames-1} frames each time)")
        
        # Create output filename
        video_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if self.preview_mode:
            # In preview mode, use temp file
            output_path = os.path.join(tempfile.gettempdir(), f"{video_name}_3d_preview_{timestamp}.mp4")
        else:
            output_path = os.path.join(output_dir, f"{video_name}_3d_{method}_{timestamp}.mp4")
        
        # Adjusted fps based on frame skipping
        output_fps = fps / skip_frames if self.preview_mode else fps
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height), isColor=True)
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Process frames
        frame_idx = start_frame
        processed_frames = 0
        start_time = time.time()
        
        # For batch processing
        batch_frames = []
        batch_indices = []
        batch_size = 4  # Process this many frames at once
        
        try:
            while frame_idx < end_frame:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video reached after processing {processed_frames} frames")
                    break
                
                # Process only selected frames
                if (frame_idx - start_frame) % skip_frames == 0:
                    # Resize frame for faster processing
                    resized_frame = self.optimize_frame(frame, resolution_scale)
                    
                    # Add to batch
                    batch_frames.append(resized_frame)
                    batch_indices.append(frame_idx)
                    
                    # Process batch when full
                    if len(batch_frames) >= batch_size or frame_idx == end_frame - 1:
                        try:
                            # Batch process depth maps
                            depth_maps = self.depth_estimator.batch_estimate_depth(batch_frames, batch_indices)
                            
                            # Process each frame in the batch
                            for i, (img, depth_map) in enumerate(zip(batch_frames, depth_maps)):
                                # Generate stereoscopic pair
                                left_view, right_view = self.vectorized_stereoscopic_pair(img, depth_map)
                                
                                # Generate output based on method
                                if method == "autostereoscopic":
                                    output_frame = self.fast_autostereoscopic(left_view, right_view)
                                elif method == "anaglyph":
                                    # Simple red-cyan anaglyph (optimized)
                                    left_mono = cv2.cvtColor(left_view, cv2.COLOR_BGR2GRAY)
                                    right_mono = cv2.cvtColor(right_view, cv2.COLOR_BGR2GRAY)
                                    
                                    output_frame = np.zeros_like(left_view)
                                    output_frame[:,:,0] = right_mono  # Blue 
                                    output_frame[:,:,1] = right_mono  # Green
                                    output_frame[:,:,2] = left_mono   # Red
                                elif method == "side_by_side":
                                    # Side-by-side (fast implementation)
                                    left_half = cv2.resize(left_view, (width//2, height))
                                    right_half = cv2.resize(right_view, (width//2, height))
                                    output_frame = np.hstack((left_half, right_half))
                                elif method == "depth_only":
                                    # Just output colorized depth map
                                    output_frame = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
                                else:
                                    output_frame = img
                                
                                out.write(output_frame)
                                processed_frames += 1
                                
                                # Save reference to last depth map for interpolation
                                self.last_depth_map = depth_map
                            
                            # Clear batch
                            batch_frames = []
                            batch_indices = []
                            
                        except Exception as e:
                            print(f"Error processing batch at frame {frame_idx}: {str(e)}")
                            # Use original frames as fallback
                            for img in batch_frames:
                                out.write(img)
                                processed_frames += 1
                            
                            # Clear batch
                            batch_frames = []
                            batch_indices = []
                
                # Update progress
                frame_idx += 1
                progress = (frame_idx - start_frame) / (end_frame - start_frame)
                progress_bar.progress(progress)
                
                # Calculate ETA
                elapsed = time.time() - start_time
                fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                remaining_frames = (end_frame - frame_idx) // skip_frames
                eta_seconds = remaining_frames / fps_processing if fps_processing > 0 else 0
                
                # Update progress text with more info
                progress_text.text(f"Processing: {processed_frames}/{frames_to_process} frames " +
                                  f"({progress:.1%}) • Speed: {fps_processing:.1f} fps • " +
                                  f"ETA: {eta_seconds//60:.0f}m {eta_seconds%60:.0f}s")
                
                # Debug info every 50 frames
                if frame_idx % 50 == 0:
                    print(f"Processed {processed_frames}/{frames_to_process} frames " +
                          f"({progress:.1%}) at {fps_processing:.1f} fps")
        
        finally:
            # Release resources
            cap.release()
            out.release()
            self.depth_estimator.release_resources()
            
            elapsed_time = time.time() - start_time
            print(f"Conversion complete in {elapsed_time:.1f} seconds.")
            print(f"Processed {processed_frames} frames at average speed of " +
                 f"{processed_frames/elapsed_time:.1f} fps")
            print(f"Output saved to: {output_path}")
            
            return output_path

# Create final UI with enhanced speed options
def get_video_download_link(video_path, filename):
    """Generate a download link for a video file"""
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">Download Processed Video</a>'
    return href

# Streamlit app with optimized performance settings
def main():
    st.set_page_config(page_title="Fast 2D to 3D Video Converter", page_icon="🎥", layout="wide")
    
    st.title("Fast 2D to 3D Video Converter")
    st.write("""
    Convert your 2D videos to autostereoscopic 3D videos that can be viewed without special glasses.
    Optimized for speed and quality balance.
    """)
    
    # Create a file uploader component for the input video
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    # Create a temp directory for output if it doesn't exist
    temp_output_dir = os.path.join(tempfile.gettempdir(), "2d_to_3d_output")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Input handling for uploaded file
    input_path = None
    if uploaded_file is not None:
        # Save uploaded file to temp directory
        input_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded successfully: {uploaded_file.name}")
    
    # Output directory 
    output_dir = temp_output_dir
    
    # Check if we have a valid input file
    if input_path and os.path.exists(input_path):
        # Display original video
        st.subheader("Original Video")
        st.video(input_path)
        
        # Get video info
        cap = cv2.VideoCapture(input_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            st.info(f"Video info: {width}x{height}, {total_frames} frames, {fps} fps, {duration:.2f} seconds")
            cap.release()
        
        # 3D conversion settings with speed options
        st.subheader("3D Conversion Settings")
        
        # Create tabs for basic and advanced settings
        basic_tab, advanced_tab = st.tabs(["Basic Settings", "Advanced Settings"])
        
        with basic_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                parallax_strength = st.slider("3D Effect Strength", 0.02, 0.15, 0.05, 0.01,
                                             help="Controls the strength of the 3D effect. Higher values create deeper 3D but may cause eye strain.")
                
                output_format = st.selectbox("Output Format", 
                                            ["autostereoscopic", "anaglyph", "side_by_side", "depth_only"],
                                            help="Autostereoscopic: for glasses-free 3D displays. Anaglyph: for red-cyan glasses. Side-by-side: for VR headsets.")
            
            with col2:
                # Speed vs quality tradeoff
                processing_speed = st.select_slider(
                    "Processing Speed",
                    options=["Highest Quality (Slow)", "Balanced", "Fast", "Very Fast", "Ultra Fast (Preview)"],
                    value="Fast",
                    help="Select processing speed vs quality tradeoff"
                )
                
                # Map selection to actual parameters
                speed_configs = {
                    "Highest Quality (Slow)": {"scale": 1.0, "skip": 1, "model": "Intel/dpt-large"},
                    "Balanced": {"scale": 0.75, "skip": 1, "model": "Intel/dpt-hybrid-midas"},
                    "Fast": {"scale": 0.5, "skip": 1, "model": "Intel/dpt-hybrid-midas"},
                    "Very Fast": {"scale": 0.5, "skip": 2, "model": "Intel/dpt-hybrid-midas"},
                    "Ultra Fast (Preview)": {"scale": 0.5, "skip": 4, "model": "Intel/dpt-hybrid-midas", "preview": True}
                }
                
                speed_config = speed_configs[processing_speed]
                
                # Show estimated processing time based on speed setting
                frame_count = total_frames
                if speed_config.get("skip", 1) > 1:
                    frame_count = frame_count // speed_config.get("skip", 1)
                    
                # Rough estimation of processing speed (frames per second)
                estimated_fps = {
                    "Highest Quality (Slow)": 0.5,
                    "Balanced": 1.5,
                    "Fast": 3,
                    "Very Fast": 6, 
                    "Ultra Fast (Preview)": 10
                }
                
                estimated_time = frame_count / estimated_fps[processing_speed]
                st.info(f"Estimated processing time: {estimated_time//60:.0f} minutes {estimated_time%60:.0f} seconds")
        
        with advanced_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                resolution_scale = st.slider("Resolution Scale", 0.25, 1.0, speed_config["scale"], 0.05,
                                           help="Scale factor for output resolution. Lower values are faster.")
                
                edge_enhancement = st.slider("Edge Enhancement", 0.0, 3.0, 0.0, 0.1,
                                            help="Enhance edges for better stereo effect. 0 = disabled.")
                
            with col2:
                skip_frames = st.slider("Process Every Nth Frame", 1, 10, speed_config.get("skip", 1),
                                       help="Process only every Nth frame to increase speed. Higher values are faster but less smooth.")
                
                model_name = st.selectbox("Depth Model", 
                                         ["Intel/dpt-hybrid-midas", "Intel/dpt-large"],
                                         index=0 if speed_config["model"] == "Intel/dpt-hybrid-midas" else 1,
                                         help="Depth estimation model to use. Hybrid is faster, Large is more accurate.")
                
                preview_mode = st.checkbox("Preview Mode", value=speed_config.get("preview", False),
                                          help="Process a subset of frames for quick preview")
                
                if preview_mode:
                    max_preview_frames = st.slider("Max Preview Frames", 10, 200, 50, 
                                                  help="Maximum number of frames to process in preview mode")
                else:
                    max_preview_frames = None
        
        # Create a section for time-range selection
        st.subheader("Time Range (Optional)")
        enable_timerange = st.checkbox("Process specific time range", value=False)
        
        if enable_timerange:
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.slider("Start Time (seconds)", 0.0, duration, 0.0, 0.5)
                start_frame = int(start_time * fps)
            with col2:
                end_time = st.slider("End Time (seconds)", 0.0, duration, duration, 0.5)
                end_frame = int(end_time * fps)
        else:
            start_frame = 0
            end_frame = total_frames
        
        # Process button
        if st.button("Convert to 3D"):
            # Show selected options
            st.info(f"Selected options: {processing_speed} mode, {output_format} format, {resolution_scale:.2f}x resolution")
            
            with st.spinner(f"Converting video to 3D... This may take several minutes."):
                try:
                    # Initialize converter with settings
                    converter = OptimizedVideo3DConverter(
                        parallax_strength=parallax_strength,
                        edge_enhancement=edge_enhancement,
                        model_name=model_name,
                        preview_mode=preview_mode
                    )
                    
                    # Process video
                    result_path = converter.convert_video(
                        input_path, 
                        output_dir, 
                        method=output_format,
                        resolution_scale=resolution_scale,
                        skip_frames=skip_frames,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        max_preview_frames=max_preview_frames
                    )
                    
                    # Display result
                    st.success("Conversion complete!")
                    st.subheader("3D Video Result")
                    st.video(result_path)
                    
                    # Get filename for download
                    output_filename = os.path.basename(result_path)
                    
                    # Download button
                    st.markdown(get_video_download_link(result_path, output_filename), unsafe_allow_html=True)
                    
                    # Tips for viewing
                    st.info("""
                    **Viewing Tips:**
                    - For autostereoscopic format, view on a glasses-free 3D display or try these techniques:
                      - Slightly cross your eyes while viewing the screen
                      - View from different angles to find the sweet spot
                      - Try different viewing distances
                    - For anaglyph format, use red-cyan 3D glasses
                    - For side-by-side format, use a VR headset or cross your eyes to merge the images
                    """)
                    
                except Exception as e:
                    st.error(f"An error occurred during conversion: {str(e)}")
    else:
        st.info("Please upload a video file to begin.")

# For direct execution
if __name__ == "__main__":
    main()
