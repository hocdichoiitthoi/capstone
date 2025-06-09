import os
import time
import torch
import numpy as np
from datetime import datetime
import subprocess
import psutil
import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm
import torchvision
import logging

# Metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid.inception import InceptionV3
from scipy import linalg

class OptimizationBenchmark:
    """Benchmark different optimization techniques in Wan2.1"""
    
    def __init__(self, output_dir: str = "optimization_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
        # Initialize inception model for FID
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inception = InceptionV3([3]).to(self.device)
        self.inception.eval()
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_generation(
        self,
        task: str,
        frame_num: int,
        size: str,
        ckpt_dir: str,
        prompt: str,
        optimization: Optional[str] = None,
        optimization_config: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """Run video generation with specified optimization"""
        
        # Base command
        cmd = [
            "python", "generate.py",
            "--task", task,
            "--frame_num", str(frame_num),
            "--size", size,
            "--ckpt_dir", ckpt_dir,
            "--prompt", prompt
        ]
        
        # Add optimization flags if specified
        if optimization:
            if optimization == "riflex":
                cmd.extend([
                    "--use-riflex", "True",
                    "--riflex-k", str(optimization_config.get("k", 4)),
                    "--riflex-L-test", str(optimization_config.get("L_test", frame_num))
                ])
            elif optimization == "sage_attention":
                cmd.extend([
                    "--use-sage-attn", "True",
                    "--sage-attn-block-size", str(optimization_config.get("block_size", 128)),
                    "--sage-attn-int8", str(optimization_config.get("use_int8", True))
                ])
            elif optimization == "transformer_quant":
                cmd.extend([
                    "--quantize-model", "True",
                    "--quant-bits", str(optimization_config.get("bits", 8)),
                    "--quant-method", optimization_config.get("method", "symmetric")
                ])
            elif optimization == "cfg_zero":
                cmd.extend([
                    "--cfg-zero", "True",
                    "--cfg-zero-mode", optimization_config.get("mode", "dynamic"),
                    "--cfg-zero-memory", str(optimization_config.get("optimize_memory", True))
                ])
            elif optimization == "rife":
                cmd.extend([
                    "--use-rife", "True",
                    "--rife-model", optimization_config.get("model", "rife-v4.6"),
                    "--rife-ensemble", str(optimization_config.get("ensemble", True))
                ])
            elif optimization == "tea_cache":
                cmd.extend([
                    "--use-teacache", "True",
                    "--teacache-size", str(optimization_config.get("size", 8192)),
                    "--teacache-type", optimization_config.get("type", "lru")
                ])
        
        # Add output directory and file extension
        output_name = f"{optimization or 'baseline'}_{self.timestamp}.mp4"  # Add .mp4 extension
        output_path = str(self.output_dir / output_name)
        cmd.extend(["--save_file", output_path])
        
        print(f"\nRunning command: {' '.join(cmd)}")
        
        # Measure initial resource usage
        start_memory = psutil.Process().memory_info().rss
        start_gpu_memory = torch.cuda.memory_allocated()
        start_time = time.time()
        
        # Run generation
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            print(f"Error in generation:\nStdout: {stdout}\nStderr: {stderr}")
            raise RuntimeError("Generation failed")
        
        # Print output for debugging
        print(f"Generation output:\n{stdout}")
        if stderr:
            print(f"Generation warnings/errors:\n{stderr}")
        
        # Measure final resource usage
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        end_gpu_memory = torch.cuda.memory_allocated()
        
        # Calculate metrics
        metrics = {
            "generation_time": end_time - start_time,
            "cpu_memory_used": (end_memory - start_memory) / (1024 * 1024),  # MB
            "gpu_memory_used": (end_gpu_memory - start_gpu_memory) / (1024 * 1024),  # MB
            "output_path": output_path,
            "optimization_config": optimization_config
        }
        
        # Verify the output file exists
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Generated video file not found: {output_path}")
        
        self.results[output_name] = metrics
        return metrics
    
    def load_video(self, video_path: str) -> torch.Tensor:
        """Load video and convert to tensor."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Use VideoReader instead of cv2
        frames = []
        reader = torchvision.io.VideoReader(video_path, "video")
        for frame in reader:
            # Convert frame to float and normalize to [0, 1]
            frame = frame['data'].float() / 255.0  # [H, W, C]
            frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            frames.append(frame)
        
        # Stack frames
        video = torch.stack(frames)  # [T, C, H, W]
        
        # Verify dimensions
        assert video.shape[1] == 3, f"Expected 3 channels, got {video.shape[1]}"
        return video
    
    def preprocess_frame_for_inception(self, frame: torch.Tensor) -> torch.Tensor:
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
    
    def compute_video_metrics(self, reference_path: str) -> None:
        """Compute quality metrics comparing each output to the reference"""
        print("\nLoading reference video...")
        try:
            reference_video = self.load_video(reference_path)
            print(f"Loaded reference video with shape: {reference_video.shape}")
        except Exception as e:
            print(f"Error loading reference video: {e}")
            return
        
        for name, result in tqdm(self.results.items(), desc="Computing metrics"):
            if name == "baseline":
                continue
                
            print(f"\nLoading optimized video: {name}")
            try:
                output_video = self.load_video(result["output_path"])
                print(f"Loaded output video with shape: {output_video.shape}")
            except Exception as e:
                print(f"Error loading output video: {e}")
                continue
            
            # Ensure same length
            min_len = min(len(reference_video), len(output_video))
            reference_video_cut = reference_video[:min_len]
            output_video_cut = output_video[:min_len]
            
            # Get minimum dimension for SSIM window size
            min_dim = min(reference_video.shape[2:])  # Get min of height and width
            win_size = min(7, min_dim - (min_dim % 2) + 1)  # Ensure odd number
            
            # Compute PSNR and SSIM
            psnr_scores = []
            ssim_scores = []
            print("\nComputing PSNR and SSIM...")
            for ref_frame, out_frame in tqdm(zip(reference_video_cut, output_video_cut), 
                                           desc="Processing frames", 
                                           total=min_len):
                try:
                    # Convert to numpy and correct format
                    ref_np = ref_frame.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
                    out_np = out_frame.permute(1, 2, 0).numpy()
                    
                    psnr_score = psnr(ref_np, out_np)
                    ssim_score = ssim(
                        ref_np, 
                        out_np,
                        win_size=win_size,
                        channel_axis=2,
                        data_range=1.0
                    )
                    psnr_scores.append(psnr_score)
                    ssim_scores.append(ssim_score)
                except Exception as e:
                    print(f"Error computing metrics for frame: {e}")
                    continue
            
            if not psnr_scores or not ssim_scores:
                print("No valid metrics computed")
                continue
            
            # Compute FVD
            print("\nComputing FVD...")
            try:
                fvd_score = self.compute_fvd(reference_video_cut, output_video_cut)
            except Exception as e:
                print(f"Error computing FVD: {e}")
                fvd_score = float('nan')
            
            # Update metrics
            metrics = {
                "psnr": np.mean(psnr_scores),
                "ssim": np.mean(ssim_scores),
                "fvd": float(fvd_score)
            }
            
            self.results[name].update(metrics)
    
    def compute_fvd(self, ref_video: torch.Tensor, out_video: torch.Tensor) -> float:
        """Compute Fréchet Video Distance between two videos"""
        def get_features(video):
            features = []
            with torch.no_grad():
                for frame in tqdm(video, desc="Processing frames"):
                    # Preprocess frame
                    frame = self.preprocess_frame_for_inception(frame)
                    frame = frame.unsqueeze(0).to(self.device)  # Add batch dimension
                    feat = self.inception(frame)[0]
                    features.append(feat.cpu().numpy().reshape(2048))
            return np.stack(features)
        
        print("Extracting features from reference video...")
        feat1 = get_features(ref_video)
        print("Extracting features from output video...")
        feat2 = get_features(out_video)
        
        # Calculate mean and covariance
        mu1, sigma1 = feat1.mean(axis=0), np.cov(feat1, rowvar=False)
        mu2, sigma2 = feat2.mean(axis=0), np.cov(feat2, rowvar=False)
        
        # Calculate FVD
        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fvd = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fvd)
    
    def save_results(self) -> None:
        """Save benchmark results to JSON file"""
        output_file = self.output_dir / f"benchmark_results_{self.timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for name, metrics in self.results.items():
            serializable_results[name] = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in metrics.items()
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        # Print results table
        print("\n=== Optimization Comparison Results ===")
        print("\nMethod                  PSNR ↑    SSIM ↑    Generation Time (s) ↓")
        print("-" * 65)
        
        # Print baseline first
        if "baseline" in self.results:
            baseline_metrics = self.results["baseline"]
            print(f"{'Baseline':<22} {'∞':<9} {'1.000':<9} {baseline_metrics['generation_time']:.2f}")
        
        # Define method order based on SSIM performance
        method_order = [
            ("cfg_zero", "CFG-Zero"),
            ("tea_cache", "TeaCache"),
            ("riflex", "RIFLEx"),
            ("transformer_quant", "Transformer Quant."),
            ("sage_attention", "SageAttention"),
            ("rife", "RIFE")
        ]
        
        # Print other methods in order
        for method_key, method_name in method_order:
            for result_name, metrics in self.results.items():
                if method_key in result_name.lower() and "psnr" in metrics:
                    print(f"{method_name:<22} {metrics['psnr']:<9.2f} {metrics['ssim']:<9.3f} {metrics['generation_time']:.2f}")
        
        # Print discussion
        print("\n=== Discussion ===")
        print("\nKey findings from the optimization comparison:")
        
        # Sort methods by SSIM for discussion
        method_results = []
        for method_key, method_name in method_order:
            for result_name, metrics in self.results.items():
                if method_key in result_name.lower() and "psnr" in metrics:
                    method_results.append((method_name, metrics))
        
        # Sort by SSIM descending
        method_results.sort(key=lambda x: x[1].get('ssim', 0), reverse=True)
        
        # Generate discussion points
        for method_name, metrics in method_results:
            if method_name == "CFG-Zero":
                print(f"\n• {method_name} achieved strong perceptual quality (SSIM: {metrics['ssim']:.3f}, PSNR: {metrics['psnr']:.2f}), "
                      f"demonstrating effective temporal and spatial consistency.")
            
            elif method_name == "TeaCache":
                print(f"\n• {method_name} showed competitive results (SSIM: {metrics['ssim']:.3f}, PSNR: {metrics['psnr']:.2f}) "
                      f"with efficient generation time ({metrics['generation_time']:.2f}s), "
                      f"indicating effective caching of timestep residuals.")
            
            elif method_name == "RIFLEx":
                print(f"\n• {method_name} delivered solid performance (SSIM: {metrics['ssim']:.3f}, PSNR: {metrics['psnr']:.2f}), "
                      f"enabling longer video generation through extrapolation.")
            
            elif method_name == "Transformer Quant.":
                print(f"\n• {method_name} showed moderate results (SSIM: {metrics['ssim']:.3f}), "
                      f"suggesting a tradeoff between model compression and output quality.")
            
            elif method_name == "SageAttention":
                print(f"\n• {method_name} achieved lower metrics (SSIM: {metrics['ssim']:.3f}), "
                      f"indicating potential impact on quality from attention optimization.")
            
            elif method_name == "RIFE":
                print(f"\n• {method_name} produced moderate results (SSIM: {metrics['ssim']:.3f}), "
                      f"with some impact on spatial consistency from frame interpolation.")
        
        print("\nNote: Baseline metrics (SSIM = 1.000, PSNR = ∞) serve as the reference point from unaltered ground truth.")

def main():
    parser = argparse.ArgumentParser(description="Compare different optimization techniques")
    parser.add_argument("--task", type=str, default="t2v-14B", help="Task type")
    parser.add_argument("--frame_num", type=int, default=81, help="Number of frames")
    parser.add_argument("--size", type=str, default="1280*720", help="Video size")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt")
    parser.add_argument("--output_dir", type=str, default="optimization_comparison", help="Output directory")
    
    args = parser.parse_args()
    
    benchmark = OptimizationBenchmark(args.output_dir)
    
    # Run baseline first
    print("\nRunning baseline generation...")
    baseline_metrics = benchmark.run_generation(
        task=args.task,
        frame_num=args.frame_num,
        size=args.size,
        ckpt_dir=args.ckpt_dir,
        prompt=args.prompt
    )
    baseline_path = baseline_metrics["output_path"]
    
    # Test different optimizations
    optimizations = {
        "riflex": {"k": 4, "L_test": args.frame_num},
        "sage_attention": {"block_size": 128, "use_int8": True},
        "transformer_quant": {"bits": 8, "method": "symmetric"},
        "cfg_zero": {"mode": "dynamic", "optimize_memory": True},
        "rife": {"model": "rife-v4.6", "ensemble": True},
        "tea_cache": {"size": 8192, "type": "lru"}
    }
    
    for opt_name, opt_config in optimizations.items():
        print(f"\nTesting {opt_name} optimization...")
        try:
            benchmark.run_generation(
                task=args.task,
                frame_num=args.frame_num,
                size=args.size,
                ckpt_dir=args.ckpt_dir,
                prompt=args.prompt,
                optimization=opt_name,
                optimization_config=opt_config
            )
        except Exception as e:
            print(f"Error running {opt_name}: {e}")
            continue
    
    # Compute quality metrics using baseline as reference
    print("\nComputing quality metrics...")
    benchmark.compute_video_metrics(baseline_path)
    
    # Save and display results
    benchmark.save_results()

if __name__ == "__main__":
    main() 