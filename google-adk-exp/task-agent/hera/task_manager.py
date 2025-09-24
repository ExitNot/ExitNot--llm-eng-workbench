"""
Task Management System for Hera Agent
Handles local storage and CRUD operations for tasks with timeframe support.
"""
# TODO: Have to be replaced with a database when tests finished.

import json
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum


class TimeFrame(Enum):
    """Time frame enumeration for task organization."""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskManager:
    """Local JSON file-based task storage and management."""

    def __init__(self, storage_file: str = "tasks.json"):
        self.storage_file = storage_file
        self.tasks = self._load_tasks()
    
    def _get_timeframe_date_range(self, timeframe: str, target_date: Optional[date] = None) -> tuple[date, date]:
        """Get the date range for a given timeframe.
        
        Args:
            timeframe: The timeframe ('day', 'week', 'month', 'year')
            target_date: The reference date (defaults to today)
            
        Returns:
            Tuple of (start_date, end_date) for the timeframe
        """
        if target_date is None:
            target_date = date.today()
            
        if timeframe.lower() == "day":
            return target_date, target_date
        elif timeframe.lower() == "week":
            # Week starts on Monday (0) and ends on Sunday (6)
            start_of_week = target_date - timedelta(days=target_date.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            return start_of_week, end_of_week
        elif timeframe.lower() == "month":
            # First day of the month
            start_of_month = target_date.replace(day=1)
            # Last day of the month
            if target_date.month == 12:
                end_of_month = target_date.replace(year=target_date.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                end_of_month = target_date.replace(month=target_date.month + 1, day=1) - timedelta(days=1)
            return start_of_month, end_of_month
        elif timeframe.lower() == "year":
            # First day of the year
            start_of_year = target_date.replace(month=1, day=1)
            # Last day of the year
            end_of_year = target_date.replace(month=12, day=31)
            return start_of_year, end_of_year
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse a date string into a date object.
        
        Args:
            date_str: Date string in ISO format (YYYY-MM-DD) or ISO datetime format
            
        Returns:
            date object or None if parsing fails
        """
        if not date_str:
            return None
            
        try:
            # Try parsing as ISO datetime first
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
            # Try parsing as ISO date
            else:
                return datetime.fromisoformat(date_str).date()
        except (ValueError, TypeError):
            return None
    
    def _calculate_due_date_from_timeframe(self, timeframe: str, reference_date: Optional[date] = None) -> str:
        """Calculate a due date based on timeframe.
        
        Args:
            timeframe: The timeframe ('day', 'week', 'month', 'year')
            reference_date: The reference date (defaults to today)
            
        Returns:
            ISO date string for the end of the timeframe period
        """
        if reference_date is None:
            reference_date = date.today()
            
        _, end_date = self._get_timeframe_date_range(timeframe, reference_date)
        return end_date.isoformat()

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from JSON file."""
        try:
            with open(self.storage_file, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_tasks(self) -> None:
        """Save tasks to JSON file."""
        with open(self.storage_file, "w") as file:
            json.dump(self.tasks, file, indent=4)

    def add_task(self, title: str, description: str = "", timeframe: str = "day", 
                 priority: str = "medium", due_date: Optional[str] = None, 
                 tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Add a new task.
        
        Args:
            title: Task title
            description: Task description
            timeframe: Task timeframe (used to calculate due_date if not provided)
            priority: Task priority
            due_date: Explicit due date (ISO format). If not provided, calculated from timeframe
            tags: List of tags
        """
        # If no due_date provided, calculate it from timeframe
        calculated_due_date = due_date
        if not due_date:
            try:
                calculated_due_date = self._calculate_due_date_from_timeframe(timeframe.lower())
            except ValueError:
                # If invalid timeframe, use today as fallback
                calculated_due_date = date.today().isoformat()
        
        task = {
            "id": str(uuid.uuid4()),
            "title": title,
            "description": description,
            "timeframe": timeframe.lower(),
            "priority": priority.lower(),
            "status": TaskStatus.PENDING.value,
            "due_date": calculated_due_date,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.tasks.append(task)
        self._save_tasks()
        
        return {
            "status": "success",
            "message": "Task added successfully",
            "task": task
        }

    def list_tasks(self, timeframe: Optional[str] = None, 
                   status: Optional[str] = None, 
                   target_date: Optional[str] = None) -> Dict[str, Any]:
        """List tasks with optional filtering.
        
        Args:
            timeframe: Filter by timeframe ('day', 'week', 'month', 'year').
                      Uses due_date as source of truth, not stored timeframe field.
            status: Filter by task status
            target_date: Reference date for timeframe filtering (ISO format, defaults to today)
        """
        filtered_tasks = self.tasks.copy()
        
        # Parse target date for timeframe filtering
        reference_date = date.today()
        if target_date:
            parsed_target = self._parse_date(target_date)
            if parsed_target:
                reference_date = parsed_target
        
        # Filter by timeframe using due_date as source of truth
        if timeframe:
            try:
                start_date, end_date = self._get_timeframe_date_range(timeframe, reference_date)
                
                def task_in_timeframe(task):
                    due_date_str = task.get("due_date")
                    if not due_date_str:
                        # If no due_date, fall back to created_at for timeframe filtering
                        created_at = task.get("created_at")
                        if created_at:
                            task_date = self._parse_date(created_at)
                        else:
                            return False
                    else:
                        task_date = self._parse_date(due_date_str)
                    
                    if task_date is None:
                        return False
                        
                    return start_date <= task_date <= end_date
                
                filtered_tasks = [t for t in filtered_tasks if task_in_timeframe(t)]
                
            except ValueError as e:
                return {
                    "status": "error",
                    "message": f"Invalid timeframe: {e}",
                    "error_code": "INVALID_TIMEFRAME"
                }
        
        # Filter by status
        if status:
            filtered_tasks = [t for t in filtered_tasks if t.get("status") == status.lower()]
        
        # Sort by due_date first (tasks with due dates), then by created_at
        def sort_key(task):
            due_date_str = task.get("due_date")
            if due_date_str:
                due_date_obj = self._parse_date(due_date_str)
                if due_date_obj:
                    # Use due_date as primary sort key, with a prefix to sort before created_at
                    return (0, due_date_obj.isoformat())
            
            # Fall back to created_at for tasks without due_date
            created_at = task.get("created_at", "")
            return (1, created_at)
        
        filtered_tasks.sort(key=sort_key)
        
        return {
            "status": "success",
            "tasks": filtered_tasks,
            "count": len(filtered_tasks),
            "filters": {
                "timeframe": timeframe,
                "status": status,
                "target_date": target_date or reference_date.isoformat()
            }
        }

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing task."""
        for task in self.tasks:
            if task["id"] == task_id:
                # Update allowed fields
                allowed_fields = ["title", "description", "timeframe", "priority", 
                                "status", "due_date", "tags"]
                
                for field, value in updates.items():
                    if field in allowed_fields:
                        task[field] = value
                
                task["updated_at"] = datetime.now().isoformat()
                self._save_tasks()
                
                return {
                    "status": "success",
                    "message": "Task updated successfully",
                    "task": task
                }
        
        return {
            "status": "error",
            "message": "Task not found",
            "error_code": "TASK_NOT_FOUND"
        }

    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """Delete a task by ID."""
        for i, task in enumerate(self.tasks):
            if task["id"] == task_id:
                deleted_task = self.tasks.pop(i)
                self._save_tasks()
                
                return {
                    "status": "success",
                    "message": "Task deleted successfully",
                    "deleted_task_id": task_id,
                    "deleted_task": deleted_task
                }
        
        return {
            "status": "error",
            "message": "Task not found",
            "error_code": "TASK_NOT_FOUND"
        }

    def get_task_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        total_tasks = len(self.tasks)
        
        status_counts = {}
        timeframe_counts = {}
        priority_counts = {}
        
        for task in self.tasks:
            status = task.get("status", "unknown")
            timeframe = task.get("timeframe", "unknown")
            priority = task.get("priority", "unknown")
            
            status_counts[status] = status_counts.get(status, 0) + 1
            timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            "status": "success",
            "stats": {
                "total_tasks": total_tasks,
                "by_status": status_counts,
                "by_timeframe": timeframe_counts,
                "by_priority": priority_counts
            }
        }
