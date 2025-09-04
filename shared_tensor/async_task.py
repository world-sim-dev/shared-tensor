"""
Async Task Management System

Provides task queue and execution management for long-running functions
that shouldn't be limited by HTTP timeouts.
"""

import uuid
import time
import pickle
import threading
import queue
import logging
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"      # Task submitted, waiting to start
    RUNNING = "running"      # Task currently executing
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"        # Task failed with error
    CANCELLED = "cancelled"  # Task was cancelled


@dataclass
class TaskInfo:
    """Task information container"""
    task_id: str
    function_path: str
    args_hex: str
    kwargs_hex: str
    options: Dict[str, Any]
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result_hex: Optional[str] = None
    error_message: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskInfo':
        """Create from dictionary"""
        data['status'] = TaskStatus(data['status'])
        return cls(**data)


class TaskExecutor:
    """Executes tasks in background threads"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_tasks: Dict[str, threading.Future] = {}
        self._shutdown = False
    
    def submit_task(self, task_info: TaskInfo, task_manager: 'TaskManager') -> bool:
        """Submit a task for execution"""
        if self._shutdown:
            return False
        
        try:
            future = self.executor.submit(self._execute_task, task_info, task_manager)
            self.running_tasks[task_info.task_id] = future
            return True
        except Exception as e:
            logger.error(f"Failed to submit task {task_info.task_id}: {e}")
            return False
    
    def _execute_task(self, task_info: TaskInfo, task_manager: 'TaskManager'):
        """Execute a single task"""
        from shared_tensor.utils import import_function_from_path, serialize_result, deserialize_args
        
        task_id = task_info.task_id
        logger.info(f"Starting task execution: {task_id}")
        
        try:
            # Update task status to running
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = time.time()
            task_manager.update_task(task_info)
            
            # Import the function
            func = import_function_from_path(task_info.function_path)
            
            # Deserialize arguments using ForkingPickler
            args = deserialize_args(task_info.args_hex)
            kwargs = deserialize_args(task_info.kwargs_hex) if task_info.kwargs_hex else {}
            
            logger.debug(f"Executing {task_info.function_path} with args={args}, kwargs={kwargs}")
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Serialize result using ForkingPickler
            result_hex = serialize_result(result).hex()
            
            # Update task with success
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = time.time()
            task_info.result_hex = result_hex
            task_manager.update_task(task_info)
            
            logger.info(f"Task completed successfully: {task_id}")
            
        except Exception as e:
            logger.error(f"Task execution failed: {task_id} - {e}")
            
            # Update task with error
            task_info.status = TaskStatus.FAILED
            task_info.completed_at = time.time()
            task_info.error_message = str(e)
            task_manager.update_task(task_info)
        
        finally:
            # Clean up
            self.running_tasks.pop(task_id, None)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        future = self.running_tasks.get(task_id)
        if future:
            return future.cancel()
        return False
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor"""
        self._shutdown = True
        self.executor.shutdown(wait=wait)


class TaskManager:
    """Manages task lifecycle and storage"""
    
    def __init__(self, max_tasks: int = 1000, cleanup_interval: int = 3600):
        self.max_tasks = max_tasks
        self.cleanup_interval = cleanup_interval
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.RLock()
        self._task_queue = queue.Queue()
        self._executor = TaskExecutor()
        self._running = False
        self._worker_thread = None
        self._cleanup_thread = None
    
    def start(self):
        """Start the task manager"""
        if self._running:
            return
        
        self._running = True
        
        # Start worker thread for processing tasks
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("TaskManager started")
    
    def stop(self):
        """Stop the task manager"""
        self._running = False
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        self._executor.shutdown(wait=True)
        logger.info("TaskManager stopped")
    
    def submit_task(self, function_path: str, args: tuple = (), kwargs: Dict[str, Any] = None, options: Dict[str, Any] = None) -> str:
        """Submit a new task for execution"""
        if kwargs is None:
            kwargs = {}
        
        # Create task info
        task_id = str(uuid.uuid4())
        
        from shared_tensor.utils import serialize_result
        
        args_hex = serialize_result(args).hex() if args else ""
        kwargs_hex = serialize_result(kwargs).hex() if kwargs else ""
        
        task_info = TaskInfo(
            task_id=task_id,
            function_path=function_path,
            args_hex=args_hex,
            kwargs_hex=kwargs_hex,
            status=TaskStatus.PENDING,
            created_at=time.time()
        )
        
        with self._lock:
            self._tasks[task_id] = task_info
            self._task_queue.put(task_id)
        
        logger.info(f"Task submitted: {task_id} - {function_path}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Get task information"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def update_task(self, task_info: TaskInfo):
        """Update task information"""
        with self._lock:
            self._tasks[task_info.task_id] = task_info
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> Dict[str, TaskInfo]:
        """List tasks, optionally filtered by status"""
        with self._lock:
            if status is None:
                return self._tasks.copy()
            else:
                return {tid: task for tid, task in self._tasks.items() 
                       if task.status == status}
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        task = self.get_task(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        # Try to cancel if running
        if task.status == TaskStatus.RUNNING:
            cancelled = self._executor.cancel_task(task_id)
        else:
            cancelled = True
        
        if cancelled:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self.update_task(task)
        
        return cancelled
    
    def get_task_result(self, task_id: str) -> Any:
        """Get task result (raises exception if not completed)"""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if task.status == TaskStatus.FAILED:
            raise RuntimeError(f"Task failed: {task.error_message}")
        
        if task.status != TaskStatus.COMPLETED:
            raise RuntimeError(f"Task not completed, current status: {task.status.value}")
        
        if task.result_hex:
            # Import deserialize function
            from .utils import deserialize_args
            
            # For consistency, we should actually use a deserialize_result function
            # but for now we'll use the existing deserialize_args
            try:
                import torch
                result_bytes = bytes.fromhex(task.result_hex)
                return torch.multiprocessing.reducer.ForkingPickler.loads(result_bytes)
            except ImportError:
                return pickle.loads(bytes.fromhex(task.result_hex))
        return None
    
    def _worker_loop(self):
        """Main worker loop for processing tasks"""
        while self._running:
            try:
                # Get next task with timeout
                try:
                    task_id = self._task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                task = self.get_task(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    continue
                
                # Submit to executor
                self._executor.submit_task(task, self)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
    
    def _cleanup_loop(self):
        """Cleanup old completed tasks"""
        while self._running:
            try:
                time.sleep(self.cleanup_interval)
                
                current_time = time.time()
                to_remove = []
                
                with self._lock:
                    for task_id, task in self._tasks.items():
                        # Remove completed tasks older than 1 hour
                        if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] 
                            and task.completed_at 
                            and current_time - task.completed_at > 3600):
                            to_remove.append(task_id)
                    
                    # Also remove if we have too many tasks
                    if len(self._tasks) > self.max_tasks:
                        # Remove oldest completed tasks
                        completed_tasks = [(tid, task) for tid, task in self._tasks.items() 
                                         if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
                        completed_tasks.sort(key=lambda x: x[1].completed_at or 0)
                        
                        excess_count = len(self._tasks) - self.max_tasks
                        for tid, _ in completed_tasks[:excess_count]:
                            to_remove.append(tid)
                    
                    # Remove tasks
                    for task_id in to_remove:
                        self._tasks.pop(task_id, None)
                
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old tasks")
                    
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")


# Global task manager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
        _task_manager.start()
    return _task_manager


def shutdown_task_manager():
    """Shutdown the global task manager"""
    global _task_manager
    if _task_manager:
        _task_manager.stop()
        _task_manager = None
