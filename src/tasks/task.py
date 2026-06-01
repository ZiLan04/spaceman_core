import sys
import torch
import logging
import abc
from pathlib import Path
from enum import Enum
from typing import Dict, Any, Optional, Tuple

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from envs.genesis_env import GenesisSim
from configs.asset_configs import *

# Setup logger for task system
def setup_task_logger(name: str, level: int = logging.INFO):
    """Setup a logger for task system"""
    logger = logging.getLogger(f"Task.{name}")
    
    if not logger.handlers:
        logger.setLevel(level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    return logger


class TaskStatus(Enum):
    """Task status enumeration"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    SUCCEED = "succeed"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


class Task(abc.ABC):
    """
    Base class for robotic manipulation tasks in PyTorch simulation environment.
    All concrete tasks should inherit from this class and implement abstract methods.
    """
    
    def __init__(
        self,
        task_name: str,
        logger: logging.Logger = None,
        num_envs: int = None,
        env_spacing=None,
        n_envs_per_row=None,
        center_envs_at_origin: bool = None,
    ):
        """
        Initialize a task
        
        Args:
            task_name: Name of the task
            logger: Logger instance for task logging
        """
        self.task_name = task_name

        self.gsim = GenesisSim(
            num_envs=num_envs,
            env_spacing=env_spacing,
            n_envs_per_row=n_envs_per_row,
            center_envs_at_origin=center_envs_at_origin,
        )
        self._scene = self.gsim.scene
        self.device = self.gsim.device
        self.datatype = self.gsim.datatype
        self.num_envs = self.gsim.num_envs
        self.env_spacing = self.gsim.env_spacing
        self.n_envs_per_row = self.gsim.n_envs_per_row
        self.center_envs_at_origin = self.gsim.center_envs_at_origin
        
        # Setup logger
        self.logger = logger if logger else setup_task_logger(task_name)
        
        # Task state management
        self.status = TaskStatus.NOT_STARTED
        self.step_count = 0
        self.max_steps = 1000  # Default maximum steps, can be overridden
        
        # Task-specific parameters
        self.task_parameters = {}
        
        # Performance tracking (minimal, as requested)
        self.total_reward = 0.0
        self.success = False
        
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the task. Should be called before starting task execution.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def step(self, dt: float = None) -> bool:
        """
        Execute one step of the task.
        
        Args:
            dt: Time step for simulation (optional)
            
        Returns:
            bool: True if step executed successfully, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def reset(self) -> bool:
        """
        Reset the task to initial state.
        
        Returns:
            bool: True if reset successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def reward(self) -> torch.Tensor:
        """
        Compute the reward for current state.
        This is the objective function for the task.
        
        Returns:
            torch.Tensor: Scalar reward value
        """
        pass
    
    def start(self) -> bool:
        """
        Start task execution.
        
        Returns:
            bool: True if task started successfully, False otherwise
        """
        if self.status == TaskStatus.RUNNING:
            self.logger.warning("Task is already running")
            return False
        
        if not self.initialize():
            self.logger.error("Task initialization failed")
            self.status = TaskStatus.FAILED
            return False
        
        self.status = TaskStatus.RUNNING
        self.step_count = 0
        self.total_reward = 0.0
        self.success = False
        
        self.logger.info(f"Task '{self.task_name}' started")
        return True
    
    def stop(self) -> bool:
        """
        Stop task execution.
        
        Returns:
            bool: True if task stopped successfully, False otherwise
        """
        if self.status in [TaskStatus.RUNNING, TaskStatus.PAUSED]:
            self.status = TaskStatus.CANCELLED
            self.logger.info(f"Task '{self.task_name}' stopped")
            return True
        return False
    
    def pause(self) -> bool:
        """
        Pause task execution.
        
        Returns:
            bool: True if task paused successfully, False otherwise
        """
        if self.status == TaskStatus.RUNNING:
            self.status = TaskStatus.PAUSED
            self.logger.info("Task paused")
            return True
        return False
    
    def resume(self) -> bool:
        """
        Resume task execution.
        
        Returns:
            bool: True if task resumed successfully, False otherwise
        """
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.RUNNING
            self.logger.info("Task resumed")
            return True
        return False
    
    def check_termination(self) -> bool:
        """
        Check if task should terminate.
        Can be overridden by specific tasks for custom termination conditions.
        
        Returns:
            bool: True if task should terminate, False otherwise
        """
        # Check if maximum steps reached
        if self.step_count >= self.max_steps:
            self.logger.info("Maximum steps reached")
            return True
        
        # Check if task is completed
        if self.status == TaskStatus.COMPLETED:
            return True
        
        # Check if task failed
        if self.status in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SUCCEED}:
            return True
        
        return False
    
    def is_success(self) -> bool:
        """
        Check if task was successful.
        Base implementation - should be overridden by specific tasks.
        
        Returns:
            bool: True if task successful, False otherwise
        """
        return self.success
    
    def get_progress(self) -> float:
        """
        Get task progress as a value between 0 and 1.
        
        Returns:
            float: Task progress (0.0 to 1.0)
        """
        if self.status == TaskStatus.COMPLETED:
            return 1.0
        elif self.status in [TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return 0.0
        else:
            return min(self.step_count / self.max_steps, 0.99)
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a status report for the task.
        
        Returns:
            Dict[str, Any]: Task status information
        """
        return {
            "task_name": self.task_name,
            "status": self.status.value,
            "step_count": self.step_count,
            "progress": self.get_progress(),
            "total_reward": self.total_reward,
            "success": self.is_success(),
        }
