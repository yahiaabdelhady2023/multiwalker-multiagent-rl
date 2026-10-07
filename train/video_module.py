import imageio.v3 as iio
import numpy as np

# Open the video file in 'write' mode using the fast PyAV plugin
def save_eval_video(video_path):
    with iio.imopen(f"{video_path}.mp4", "w", plugin="pyav") as vid:
        
        # Initialize the video stream (libx264 is the standard MP4 codec)
        vid.init_video_stream("libx264", fps=30)
        
        # Simulate a loop (e.g., your environment steps)
        for _ in range(100):
            # Grab your frame (must be a Numpy array of uint8)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Write it directly to the disk
            vid.write_frame(frame)
            
    print("Video saved successfully!")