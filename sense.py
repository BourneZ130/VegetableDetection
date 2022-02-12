import pyrealsense2 as rs
import numpy as np
import cv2
import time


def get_frame():
    # Create a pipeline
    pipeline = rs.pipeline()
    

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    time.sleep(1)
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            
            decimation = rs.decimation_filter()
            decimation.set_option(rs.option.filter_magnitude, 1)

            depth_to_disparity = rs.disparity_transform(True)
            disparity_to_depth = rs.disparity_transform(False)      

            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 3)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.8)
            spatial.set_option(rs.option.filter_smooth_delta, 8)
            spatial.set_option(rs.option.holes_fill, 4)

            hole_filling = rs.hole_filling_filter()
        
            filled_depth = decimation.process(aligned_depth_frame)
            filled_depth = depth_to_disparity.process(filled_depth)
            filled_depth = spatial.process(filled_depth)
            filled_depth = disparity_to_depth.process(filled_depth)
            filled_depth = hole_filling.process(filled_depth)
            if not aligned_depth_frame or not color_frame:
                continue
            else:
                depth_image = np.asanyarray(filled_depth.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                ori_depth = np.asanyarray(aligned_depth_frame.get_data())
                break
    finally:
        pipeline.stop()
    return color_image,depth_image,ori_depth
