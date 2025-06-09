from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
import shutil
import os
from typing import Optional, Dict
from datetime import datetime
import argparse
import json
import logging
import asyncio
from uuid import uuid4

from generate import generate, _validate_args, WAN_CONFIGS, SUPPORTED_SIZES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")

# Model checkpoint directories
MODEL_DIRS = {
    't2v-1.3B': 'Wan2.1-T2V-1.3B',
    't2v-14B': 'Wan2.1-T2V-14B',
    't2i-14B': 'Wan2.1-T2V-14B'  # Using same directory as t2v-14B for text to image
}

# Default generation parameters
DEFAULT_PARAMS = {
    'frame_num': 81,  # Default FPS for video
    'size': '832*480',  # Default size
    'sample_shift': 8.0,  # Default sample shift
    'sample_guide_scale': 5.0,  # Default guide scale
    'prompt': "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",  # Default prompt for video
    't2i_prompt': "A photorealistic portrait of a majestic white cat wearing sunglasses, sitting regally against a vibrant sunset background.",  # Default prompt for image
    'sample_steps': {  # Task-specific sample steps
        't2v': 50,
        't2i': 30
    }
}

# Available tasks
AVAILABLE_TASKS = ['t2v-1.3B', 't2v-14B', 't2i-14B']

def get_checkpoint_dir(task: str) -> Path:
    """Get the checkpoint directory for a specific task"""
    if task not in MODEL_DIRS:
        raise ValueError(f"Unknown task: {task}")
    return Path(MODEL_DIRS[task])

def get_required_checkpoints(task: str) -> list:
    """Get list of required checkpoint paths for a task"""
    if task not in WAN_CONFIGS:
        return []
    
    config = WAN_CONFIGS[task]
    checkpoint_paths = []
    
    # T5 checkpoint
    if hasattr(config, 't5_checkpoint'):
        checkpoint_paths.append(config.t5_checkpoint)
    
    # VAE checkpoint
    if hasattr(config, 'vae_checkpoint'):
        checkpoint_paths.append(config.vae_checkpoint)
    
    return checkpoint_paths

def validate_checkpoints(task: str) -> bool:
    """Validate that all required checkpoint files exist for the given task"""
    if task not in WAN_CONFIGS:
        return False
    
    checkpoint_dir = get_checkpoint_dir(task)
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    # Check if all checkpoint files exist
    for checkpoint in get_required_checkpoints(task):
        checkpoint_path = checkpoint_dir / checkpoint
        if not checkpoint_path.exists():
            logger.warning(f"Missing checkpoint file: {checkpoint_path}")
            return False
    return True

app = FastAPI(title="Wan Video Generation UI")

# Create all required directories
for directory in [OUTPUT_DIR, STATIC_DIR, TEMPLATES_DIR]:
    directory.mkdir(exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class TaskStatus:
    def __init__(self):
        self.status = "queued"
        self.progress = 0
        self.total_steps = 0
        self.start_time = datetime.now().isoformat()
        self.output_file = None
        self.error = None

    def to_dict(self):
        return {
            "status": self.status,
            "progress": self.progress,
            "total_steps": self.total_steps,
            "start_time": self.start_time,
            "output_file": self.output_file,
            "error": self.error
        }

# Store task status
tasks_status: Dict[str, TaskStatus] = {}

class WebArgs:
    """Class to mimic argparse.Namespace for compatibility with generate function"""
    def __init__(self, **kwargs):
        # Default values for optimization parameters
        self.use_riflex = False
        self.riflex_k = None
        self.riflex_L_test = None
        self.use_sage_attn = False
        self.sage_attn_block_size = 128
        self.sage_attn_int8 = True
        self.quantize_model = False
        self.quant_bits = 8
        self.quant_method = "symmetric"
        self.cfg_zero = False
        self.cfg_zero_mode = "dynamic"
        self.cfg_zero_memory = True
        self.use_teacache = False
        self.teacache_size = 8192
        self.teacache_type = "lru"
        
        # Set provided values
        for key, value in kwargs.items():
            setattr(self, key, value)

def safe_file_cleanup(file_path: Path) -> None:
    """Safely cleanup a file with proper error handling"""
    try:
        if file_path and file_path.exists():
            file_path.chmod(0o666)  # Ensure we have write permission
            file_path.unlink()
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

async def generate_video_task(task_id: str, args: WebArgs):
    """Background task for video generation"""
    try:
        # Validate checkpoints before starting generation
        if not validate_checkpoints(args.task):
            required_files = get_required_checkpoints(args.task)
            checkpoint_dir = get_checkpoint_dir(args.task)
            raise ValueError(
                f"Missing required checkpoint files for {args.task}. "
                f"Please ensure all required files exist in {checkpoint_dir}: {', '.join(required_files)}"
            )

        tasks_status[task_id].status = "processing"
        tasks_status[task_id].total_steps = args.sample_steps

        # Create a progress callback
        def update_progress(step: int):
            tasks_status[task_id].progress = step

        # Modify generate function to accept progress callback
        args.progress_callback = update_progress
        generate(args)
        
        output_file = Path(args.save_file)
        if output_file.exists():
            tasks_status[task_id].status = "completed"
            tasks_status[task_id].output_file = str(output_file)
            tasks_status[task_id].progress = tasks_status[task_id].total_steps
        else:
            tasks_status[task_id].status = "failed"
            tasks_status[task_id].error = "Output file was not created"
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        tasks_status[task_id].status = "failed"
        tasks_status[task_id].error = str(e)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    # Only show available tasks
    tasks = [task for task in AVAILABLE_TASKS if task in WAN_CONFIGS]
    # Check which models have their checkpoints available
    available_models = {task: validate_checkpoints(task) for task in tasks}
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tasks": tasks,
            "sizes": list(SUPPORTED_SIZES.values()),
            "available_models": available_models,
            "default_params": DEFAULT_PARAMS
        }
    )

@app.post("/generate")
async def start_generation(
    background_tasks: BackgroundTasks,
    task: str = Form(...),
    size: str = Form(DEFAULT_PARAMS['size']),
    prompt: str = Form(DEFAULT_PARAMS['prompt']),
    frame_num: Optional[int] = Form(DEFAULT_PARAMS['frame_num']),
    sample_steps: Optional[int] = Form(None),
    sample_shift: Optional[float] = Form(DEFAULT_PARAMS['sample_shift']),
    sample_guide_scale: Optional[float] = Form(DEFAULT_PARAMS['sample_guide_scale']),
    base_seed: int = Form(-1),
    # Optional optimization parameters
    use_riflex: bool = Form(False),
    riflex_k: Optional[int] = Form(None),
    riflex_L_test: Optional[int] = Form(None),
    use_sage_attn: bool = Form(False),
    sage_attn_block_size: int = Form(128),
    sage_attn_int8: bool = Form(True),
    quantize_model: bool = Form(False),
    quant_bits: int = Form(8),
    quant_method: str = Form("symmetric"),
    cfg_zero: bool = Form(False),
    cfg_zero_mode: str = Form("dynamic"),
    cfg_zero_memory: bool = Form(True),
    use_teacache: bool = Form(False),
    teacache_size: int = Form(8192),
    teacache_type: str = Form("lru")
):
    """Start generation as a background task"""
    # Validate task
    if task not in AVAILABLE_TASKS:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid task: {task}. Available tasks are: {', '.join(AVAILABLE_TASKS)}"}
        )

    # Validate checkpoints first
    if not validate_checkpoints(task):
        checkpoint_dir = get_checkpoint_dir(task)
        required_files = get_required_checkpoints(task)
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"Required checkpoint files for {task} not found in {checkpoint_dir}. "
                f"Required files: {', '.join(required_files)}"
            }
        )

    task_id = str(uuid4())
    output_file = None
    
    try:
        # Create output filename
        output_file = OUTPUT_DIR / f"output_{task_id}.mp4"

        # Create args object compatible with generate function
        args = WebArgs(
            # Basic parameters
            task=task,
            size=size,
            prompt=prompt,
            frame_num=frame_num if task.startswith('t2v') else None,
            sample_steps=sample_steps,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            base_seed=base_seed,
            
            # Default values for other required args
            ckpt_dir=str(get_checkpoint_dir(task)),
            offload_model=True,
            ulysses_size=1,
            ring_size=1,
            t5_fsdp=False,
            t5_cpu=False,
            dit_fsdp=False,
            sample_solver="unipc",
            
            # Optimization parameters
            use_riflex=use_riflex,
            riflex_k=riflex_k,
            riflex_L_test=riflex_L_test,
            use_sage_attn=use_sage_attn,
            sage_attn_block_size=sage_attn_block_size,
            sage_attn_int8=sage_attn_int8,
            quantize_model=quantize_model,
            quant_bits=quant_bits,
            quant_method=quant_method,
            cfg_zero=cfg_zero,
            cfg_zero_mode=cfg_zero_mode,
            cfg_zero_memory=cfg_zero_memory,
            use_teacache=use_teacache,
            teacache_size=teacache_size,
            teacache_type=teacache_type,
            
            # Set output path
            save_file=str(output_file)
        )

        # Validate args
        try:
            _validate_args(args)
        except AssertionError as e:
            return JSONResponse(
                status_code=400,
                content={"detail": str(e)}
            )

        # Initialize task status
        tasks_status[task_id] = TaskStatus()

        # Start generation in background
        background_tasks.add_task(generate_video_task, task_id, args)

        return JSONResponse({
            "task_id": task_id,
            "status": "queued",
            "message": "Generation started"
        })

    except Exception as e:
        logger.error(f"Unexpected error in generate: {str(e)}")
        if task_id in tasks_status:
            tasks_status[task_id].status = "failed"
            tasks_status[task_id].error = str(e)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected error: {str(e)}"}
        )

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get the status of a video generation task"""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_status[task_id].to_dict()

@app.get("/video/{task_id}")
async def get_video(task_id: str):
    """Get the generated video if it's ready"""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = tasks_status[task_id]
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Video not ready. Current status: {task_info['status']}")
    
    output_file = Path(task_info["output_file"])
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        str(output_file),
        media_type="video/mp4",
        filename=output_file.name
    )

@app.get("/checkpoints/{task}")
async def check_checkpoints(task: str):
    """Check if checkpoints are available for a specific task"""
    if task not in WAN_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task}")
    
    required_files = get_required_checkpoints(task)
    missing_files = [
        f for f in required_files
        if not Path(f).exists()
    ]
    
    return {
        "task": task,
        "checkpoints_available": len(missing_files) == 0,
        "missing_files": missing_files,
        "required_files": required_files
    }

if __name__ == "__main__":
    # Log checkpoint information on startup
    logger.info("Checking model checkpoints...")
    
    for task in WAN_CONFIGS:
        checkpoint_dir = get_checkpoint_dir(task)
        required_files = get_required_checkpoints(task)
        if validate_checkpoints(task):
            logger.info(f"Checkpoints for {task} are available")
            logger.info(f"Using checkpoint directory: {checkpoint_dir.absolute()}")
            logger.info(f"Using checkpoint files: {', '.join(required_files)}")
        else:
            logger.warning(f"Checkpoints for {task} are missing")
            logger.warning(f"Required files: {', '.join(required_files)}")
            logger.warning(f"Please place the checkpoint files in: {checkpoint_dir.absolute()}")
            
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 