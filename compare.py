import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import psutil
import GPUtil
from tqdm import tqdm

def is_video_file(file_path: str) -> bool:
    """Check if file is a video based on extension."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    return os.path.splitext(file_path)[1].lower() in video_extensions

def load_image(image_path: str) -> torch.Tensor:
    """Load image and convert to tensor."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    transform = torchvision.transforms.ToTensor()
    return transform(image).unsqueeze(0)  # Add time dimension to match video format [1, C, H, W]

def load_video(video_path: str) -> torch.Tensor:
    """Load video and convert to tensor."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Use VideoReader instead of read_video
    frames = []
    reader = torchvision.io.VideoReader(video_path, "video")
    for frame in reader:
        # Convert frame to float and normalize to [0, 1]
        frame = frame['data'].float() / 255.0  # [H, W, C]
        # Ensure frame is in [C, H, W] format
        if frame.shape[-1] == 3:  # If last dimension is channels
            frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        frames.append(frame)
    
    # Stack frames
    video = torch.stack(frames)  # [T, C, H, W]
    
    # Verify dimensions
    assert video.shape[1] == 3, f"Expected 3 channels, got {video.shape[1]}"
    return video

def load_media(path: str) -> Tuple[torch.Tensor, bool]:
    """Load either image or video and return with type flag."""
    is_video = is_video_file(path)
    if is_video:
        return load_video(path), True
    else:
        return load_image(path), False

def calculate_psnr(media1: torch.Tensor, media2: torch.Tensor) -> float:
    """Calculate PSNR between two media files."""
    if media1.shape != media2.shape:
        raise ValueError(f"Media must have same shape. Got {media1.shape} and {media2.shape}")
    
    psnr_values = []
    for frame1, frame2 in zip(media1, media2):
        frame1_np = frame1.permute(1, 2, 0).numpy()
        frame2_np = frame2.permute(1, 2, 0).numpy()
        psnr_values.append(psnr(frame1_np, frame2_np))
    
    return np.mean(psnr_values)

def calculate_ssim(media1: torch.Tensor, media2: torch.Tensor) -> float:
    """Calculate SSIM between two media files."""
    if media1.shape != media2.shape:
        raise ValueError(f"Media must have same shape. Got {media1.shape} and {media2.shape}")
    
    ssim_values = []
    # Get minimum dimension to determine window size
    min_dim = min(media1.shape[2:])  # Get min of height and width
    # Choose window size based on image size (must be odd and <= min dimension)
    win_size = min(7, min_dim - (min_dim % 2) + 1)  # Ensure odd number
    
    for frame1, frame2 in zip(media1, media2):
        frame1_np = frame1.permute(1, 2, 0).numpy()
        frame2_np = frame2.permute(1, 2, 0).numpy()
        try:
            ssim_val = ssim(
                frame1_np, 
                frame2_np, 
                win_size=win_size,
                channel_axis=2,  # Specify channel axis explicitly
                data_range=1.0   # Since we're working with normalized data
            )
            ssim_values.append(ssim_val)
        except Exception as e:
            logging.error(f"Error calculating SSIM: {str(e)}")
            raise
    
    return np.mean(ssim_values)

def calculate_fid(media1: torch.Tensor, media2: torch.Tensor) -> float:
    """Calculate Fréchet Inception Distance between two media files."""
    try:
        from pytorch_fid.inception import InceptionV3
        from scipy import linalg
    except ImportError:
        print("Please install pytorch-fid: pip install pytorch-fid")
        return -1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize inception model
    inception = InceptionV3([3]).to(device)
    inception.eval()
    
    def preprocess_frame(frame):
        """Preprocess a single frame for inception model."""
        # Ensure frame is in [C, H, W] format
        if frame.shape[0] != 3:
            frame = frame.permute(2, 0, 1)
        
        # Resize and center crop
        resize = torchvision.transforms.Resize(299, antialias=True)
        center_crop = torchvision.transforms.CenterCrop(299)
        frame = center_crop(resize(frame))
        
        # Normalize
        frame = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(frame)
        
        return frame
    
    def get_features(media):
        features = []
        with torch.no_grad():
            for frame in tqdm(media, desc="Processing frames"):
                # Preprocess frame
                frame = preprocess_frame(frame)
                frame = frame.unsqueeze(0).to(device)  # Add batch dimension
                feat = inception(frame)[0]
                features.append(feat.cpu().numpy().reshape(2048))
        return np.stack(features)
    
    print("Extracting features from first video...")
    feat1 = get_features(media1)
    print("Extracting features from second video...")
    feat2 = get_features(media2)
    
    # Calculate mean and covariance
    mu1, sigma1 = feat1.mean(axis=0), np.cov(feat1, rowvar=False)
    mu2, sigma2 = feat2.mean(axis=0), np.cov(feat2, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return float(fid)

def get_media_info(path: str, is_video: bool) -> Dict[str, any]:
    """Get media file information."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    info = {
        "file_size_mb": os.path.getsize(path) / (1024 * 1024)
    }
    
    if is_video:
        try:
            # Use VideoReader for video metadata
            reader = torchvision.io.VideoReader(path, "video")
            metadata = reader.get_metadata()
            
            # Count frames by iterating through the video with progress bar
            print(f"Counting frames in {os.path.basename(path)}...")
            frame_count = sum(1 for _ in tqdm(reader))
            
            # Get first frame to determine resolution
            reader = torchvision.io.VideoReader(path, "video")  # Reset reader
            first_frame = next(reader)['data']
            height, width = first_frame.shape[:2]
            
            info.update({
                "fps": float(metadata["video"]["fps"][0]) if isinstance(metadata["video"]["fps"], list) else float(metadata["video"]["fps"]),
                "duration": float(metadata["video"]["duration"][0]) if isinstance(metadata["video"]["duration"], list) else float(metadata["video"]["duration"]),
                "frame_count": frame_count,
                "resolution": f"{width}x{height}"
            })
        except Exception as e:
            logging.error(f"Error reading video metadata: {str(e)}")
            raise
    else:
        img = Image.open(path)
        info.update({
            "resolution": f"{img.width}x{img.height}",
            "mode": img.mode,
            "format": img.format
        })
    
    return info

def get_system_metrics() -> Dict[str, float]:
    """Get system metrics (CPU, RAM, GPU usage)."""
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "gpu_info": []
    }
    
    # Get GPU information if available
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            metrics["gpu_info"].append({
                "id": gpu.id,
                "name": gpu.name,
                "memory_used_percent": gpu.memoryUtil * 100,
                "gpu_load_percent": gpu.load * 100
            })
    except:
        pass
    
    return metrics

def compare_media(path1: str, path2: str) -> Dict[str, any]:
    """Compare two media files (images or videos) and return metrics."""
    print(f"Loading media files...")
    media1, is_video1 = load_media(path1)
    media2, is_video2 = load_media(path2)
    
    if is_video1 != is_video2:
        raise ValueError("Both files must be of the same type (either both videos or both images)")
    
    metrics = {
        "media_type": "video" if is_video1 else "image"
    }
    
    # Get media information
    print(f"Getting media information...")
    metrics["file1_info"] = get_media_info(path1, is_video1)
    metrics["file2_info"] = get_media_info(path2, is_video2)
    
    # Calculate quality metrics
    print(f"Calculating PSNR...")
    metrics["psnr"] = calculate_psnr(media1, media2)
    
    print(f"Calculating SSIM...")
    metrics["ssim"] = calculate_ssim(media1, media2)
    
    print(f"Calculating {'FVD' if is_video1 else 'FID'}...")
    metrics["distance"] = calculate_fid(media1, media2)
    metrics["distance_type"] = "FVD" if is_video1 else "FID"
    
    # Get system metrics
    metrics["system_metrics"] = get_system_metrics()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare two media files (images or videos) using various metrics")
    parser.add_argument("file1", type=str, help="Path to first file")
    parser.add_argument("file2", type=str, help="Path to second file")
    parser.add_argument("--output", type=str, default="comparison_results.txt", help="Output file for results")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )
    
    try:
        start_time = time.time()
        metrics = compare_media(args.file1, args.file2)
        end_time = time.time()
        
        # Add comparison time
        metrics["comparison_time"] = end_time - start_time
        
        # Print results
        print("\n=== Comparison Results ===")
        print(f"\nMedia Type: {metrics['media_type'].capitalize()}")
        
        print(f"\nFile 1 ({os.path.basename(args.file1)}):")
        for k, v in metrics["file1_info"].items():
            print(f"  {k}: {v}")
        
        print(f"\nFile 2 ({os.path.basename(args.file2)}):")
        for k, v in metrics["file2_info"].items():
            print(f"  {k}: {v}")
        
        print("\nQuality Metrics:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  {metrics['distance_type']}: {metrics['distance']:.2f}")
        
        print("\nSystem Metrics:")
        print(f"  CPU Usage: {metrics['system_metrics']['cpu_percent']}%")
        print(f"  RAM Usage: {metrics['system_metrics']['ram_percent']}%")
        for gpu in metrics['system_metrics']['gpu_info']:
            print(f"  GPU {gpu['id']} ({gpu['name']}):")
            print(f"    Memory Usage: {gpu['memory_used_percent']:.1f}%")
            print(f"    GPU Load: {gpu['gpu_load_percent']:.1f}%")
        
        print(f"\nComparison Time: {metrics['comparison_time']:.2f} seconds")
        
        # Save results to file
        with open(args.output, 'w') as f:
            f.write("=== Comparison Results ===\n")
            f.write(str(metrics))
        
        print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        logging.error(f"Error during comparison: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 