"""
Hera - Google ADK Task Management Agent

This agent provides comprehensive task management capabilities with support for 
different timeframes (day, week, month, year) and returns structured JSON responses
compatible with A2A (Agent-to-Agent) communication.
"""

from google.adk.agents import Agent
from typing import Dict, Any, List, Optional
from datetime import date, timedelta, datetime
import re
from dateutil.parser import parse as dateutil_parse
from dateutil.relativedelta import relativedelta

from hera.task_manager import TaskManager

# Initialize the task manager
task_manager = TaskManager()


def parse_date_expression(expression: str) -> Dict[str, Any]:
    """
    Parse natural language date expressions and return structured date information.
    
    This function can handle various natural language expressions like:
    - "now", "today" → current date/time
    - "tomorrow" → tomorrow's date
    - "next Friday", "next Monday" → next occurrence of that weekday
    - "next week", "next month", "next year" → appropriate future dates
    - "in 3 days", "in 2 weeks" → relative dates
    - ISO dates like "2024-12-25" → parsed and validated
    
    Args:
        expression (str): Natural language date expression or ISO date string
        
    Returns:
        dict: A structured response containing:
            - status (str): Either 'success' or 'error'
            - date (str): Parsed date in ISO format (YYYY-MM-DD) (only on success)
            - datetime (str): Parsed datetime in ISO format (only on success)
            - suggested_timeframe (str): Suggested timeframe based on the expression (only on success)
            - message (str): Human-readable status message
            - error_code (str): Error code for programmatic handling (only on error)
    """
    try:
        if not expression or not expression.strip():
            return {
                "status": "error",
                "message": "Empty date expression provided",
                "error_code": "EMPTY_EXPRESSION"
            }
        
        expression = expression.strip().lower()
        today = date.today()
        now = datetime.now()
        
        # Handle current time/date expressions
        if expression in ["now", "current time", "current date and time"]:
            return {
                "status": "success",
                "date": today.isoformat(),
                "datetime": now.isoformat(),
                "suggested_timeframe": "day",
                "message": f"Current date and time: {now.isoformat()}"
            }
        
        if expression in ["today"]:
            return {
                "status": "success",
                "date": today.isoformat(),
                "datetime": today.isoformat(),
                "suggested_timeframe": "day",
                "message": f"Today: {today.isoformat()}"
            }
        
        # Handle tomorrow
        if expression in ["tomorrow"]:
            tomorrow = today + timedelta(days=1)
            return {
                "status": "success",
                "date": tomorrow.isoformat(),
                "datetime": tomorrow.isoformat(),
                "suggested_timeframe": "day",
                "message": f"Tomorrow: {tomorrow.isoformat()}"
            }
        
        # Handle next weekday (next Monday, next Friday, etc.)
        weekdays = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        
        next_weekday_match = re.match(r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", expression)
        if next_weekday_match:
            target_weekday = weekdays[next_weekday_match.group(1)]
            days_ahead = target_weekday - today.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            target_date = today + timedelta(days=days_ahead)
            return {
                "status": "success",
                "date": target_date.isoformat(),
                "datetime": target_date.isoformat(),
                "suggested_timeframe": "day",
                "message": f"Next {next_weekday_match.group(1).title()}: {target_date.isoformat()}"
            }
        
        # Handle relative time expressions
        if expression in ["next week"]:
            next_week = today + timedelta(days=7)
            return {
                "status": "success",
                "date": next_week.isoformat(),
                "datetime": next_week.isoformat(),
                "suggested_timeframe": "week",
                "message": f"Next week: {next_week.isoformat()}"
            }
        
        if expression in ["next month"]:
            next_month = today + relativedelta(months=1)
            return {
                "status": "success",
                "date": next_month.isoformat(),
                "datetime": next_month.isoformat(),
                "suggested_timeframe": "month",
                "message": f"Next month: {next_month.isoformat()}"
            }
        
        if expression in ["next year"]:
            next_year = today + relativedelta(years=1)
            return {
                "status": "success",
                "date": next_year.isoformat(),
                "datetime": next_year.isoformat(),
                "suggested_timeframe": "year",
                "message": f"Next year: {next_year.isoformat()}"
            }
        
        # Handle "in X days/weeks/months" expressions
        relative_match = re.match(r"in\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)", expression)
        if relative_match:
            amount = int(relative_match.group(1))
            unit = relative_match.group(2)
            
            if unit.startswith("day"):
                target_date = today + timedelta(days=amount)
                suggested_timeframe = "day"
            elif unit.startswith("week"):
                target_date = today + timedelta(weeks=amount)
                suggested_timeframe = "week" if amount == 1 else "month"
            elif unit.startswith("month"):
                target_date = today + relativedelta(months=amount)
                suggested_timeframe = "month"
            elif unit.startswith("year"):
                target_date = today + relativedelta(years=amount)
                suggested_timeframe = "year"
            
            return {
                "status": "success",
                "date": target_date.isoformat(),
                "datetime": target_date.isoformat(),
                "suggested_timeframe": suggested_timeframe,
                "message": f"In {amount} {unit}: {target_date.isoformat()}"
            }
        
        # Try to parse as ISO date or other standard formats
        try:
            parsed_date = dateutil_parse(expression, default=now)
            # If it's just a date (no time), use the date part
            if parsed_date.hour == 0 and parsed_date.minute == 0 and parsed_date.second == 0:
                date_only = parsed_date.date()
                return {
                    "status": "success",
                    "date": date_only.isoformat(),
                    "datetime": date_only.isoformat(),
                    "suggested_timeframe": "day",
                    "message": f"Parsed date: {date_only.isoformat()}"
                }
            else:
                return {
                    "status": "success",
                    "date": parsed_date.date().isoformat(),
                    "datetime": parsed_date.isoformat(),
                    "suggested_timeframe": "day",
                    "message": f"Parsed datetime: {parsed_date.isoformat()}"
                }
        except (ValueError, TypeError):
            pass
        
        # If we get here, we couldn't parse the expression
        return {
            "status": "error",
            "message": f"Could not parse date expression: '{expression}'. Try expressions like 'tomorrow', 'next Friday', 'in 3 days', or ISO dates like '2024-12-25'",
            "error_code": "UNPARSEABLE_EXPRESSION"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error parsing date expression: {str(e)}",
            "error_code": "PARSING_ERROR"
        }

def add_task(title: str, description: str, timeframe: str, 
             priority: str, due_date: Optional[str], 
             tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Add a new task to the task management system.
    
    IMPORTANT: Before calling this function, you MUST call parse_date_expression first
    to get the due_date and suggested_timeframe from any user date expression.
    
    WORKFLOW:
    1. User mentions a date → call parse_date_expression(user_date_expression)
    2. Get result["date"] → use as due_date parameter
    3. Get result["suggested_timeframe"] → use as timeframe parameter
    4. Then call add_task with the parsed values

    Args:
        title (str): The task title. This is a required field that describes what needs to be done.
        description (str): A detailed description of the task. Generate this based on title and context or leave empty if no context.
        timeframe (str): Use the "suggested_timeframe" from parse_date_expression result. Must be one of: 'day', 'week', 'month', 'year'.
        priority (str): The task priority level. Must be one of: 'low', 'medium', 'high'.
        due_date (str, optional): Use the "date" from parse_date_expression result (ISO format YYYY-MM-DD).
        tags (List[str], optional): A list of tags to categorize and organize the task. Pass None if not needed.
    
    Returns:
        dict: A structured response containing the task creation result:
            - status (str): Either 'success' or 'error'
            - message (str): Human-readable status message
            - task (dict): The created task object (only on success)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    try:
        # Handle defaults inside the function
        if not description:
            description = ""
        if not timeframe:
            timeframe = "day"
        if not priority:
            priority = "medium"
        if tags is None:
            tags = []
        
        # Validate timeframe
        valid_timeframes = ["day", "week", "month", "year"]
        if timeframe.lower() not in valid_timeframes:
            return {
                "status": "error",
                "message": f"Invalid timeframe. Must be one of: {valid_timeframes}",
                "error_code": "INVALID_TIMEFRAME"
            }
        
        # Validate priority
        valid_priorities = ["low", "medium", "high"]
        if priority.lower() not in valid_priorities:
            return {
                "status": "error",
                "message": f"Invalid priority. Must be one of: {valid_priorities}",
                "error_code": "INVALID_PRIORITY"
            }
        
        return task_manager.add_task(title, description, timeframe, priority, due_date, tags)
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to add task: {str(e)}",
            "error_code": "INTERNAL_ERROR"
        }


def list_tasks(timeframe: Optional[str], status: Optional[str], 
               target_date: Optional[str], include_stats: bool) -> Dict[str, Any]:
    """
    List tasks with optional filtering by timeframe, status, and target date.
    
    This function retrieves tasks from the task management system with optional
    filtering capabilities and can include statistical information about the tasks.
    
    Args:
        timeframe (str, optional): Filter tasks by timeframe. Must be one of: 'day', 'week', 'month', 'year' or None (no filtering).
        status (str, optional): Filter tasks by status. Must be one of: 'pending', 'in_progress', 'completed', 'cancelled' or None (no filtering).
        target_date (str, optional): Reference date for timeframe filtering (ISO format YYYY-MM-DD) or None for today.
        include_stats (bool): Whether to include task statistics in the response.
    
    Returns:
        dict: A structured response containing the task listing result:
            - status (str): Either 'success' or 'error'
            - tasks (list): List of task objects matching the filters (only on success)
            - count (int): Number of tasks returned (only on success)
            - filters (dict): Applied filters for reference (only on success)
            - stats (dict): Task statistics if include_stats is True (optional)
            - message (str): Human-readable status message (only on error)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    try:
        # Handle None string values that might come from the agent
        if timeframe in ["None", ""]:
            timeframe = None
        if status in ["None", ""]:
            status = None
        if target_date in ["None", ""]:
            target_date = None
        
        # Validate filters if provided
        if timeframe:
            valid_timeframes = ["day", "week", "month", "year"]
            if timeframe.lower() not in valid_timeframes:
                return {
                    "status": "error",
                    "message": f"Invalid timeframe. Must be one of: {valid_timeframes}",
                    "error_code": "INVALID_TIMEFRAME"
                }
        
        if status:
            valid_statuses = ["pending", "in_progress", "completed", "cancelled"]
            if status.lower() not in valid_statuses:
                return {
                    "status": "error",
                    "message": f"Invalid status. Must be one of: {valid_statuses}",
                    "error_code": "INVALID_STATUS"
                }
        result = task_manager.list_tasks(timeframe, status, target_date)

        if include_stats:
            stats_result = task_manager.get_task_stats()
            if stats_result["status"] == "success":
                result["stats"] = stats_result["stats"]
        
        return result
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list tasks: {str(e)}",
            "error_code": "INTERNAL_ERROR"
        }


def list_tomorrow_tasks(status: Optional[str]) -> Dict[str, Any]:
    """
    List tasks scheduled for tomorrow based on their due dates.
    
    This function retrieves all tasks that are due tomorrow, using the due_date field
    as the source of truth rather than the stored timeframe field. Tasks without
    due dates are excluded from the results.
    
    Args:
        status (str, optional): Filter tasks by status. Must be one of: 'pending', 'in_progress', 'completed', 'cancelled' or None (no filtering).
    
    Returns:
        dict: A structured response containing tomorrow's tasks:
            - status (str): Either 'success' or 'error'
            - tasks (list): List of task objects due tomorrow (only on success)
            - count (int): Number of tasks returned (only on success)
            - filters (dict): Applied filters for reference (only on success)
            - message (str): Human-readable status message (only on error)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    tomorrow = date.today() + timedelta(days=1)
    return list_tasks(timeframe="day", status=status, target_date=tomorrow.isoformat(), include_stats=False)


def list_next_week_tasks(status: Optional[str]) -> Dict[str, Any]:
    """
    List tasks scheduled for next week based on their due dates.
    
    This function retrieves all tasks that are due during next week (Monday to Sunday),
    using the due_date field as the source of truth rather than the stored timeframe field.
    The week calculation is based on a 7-day offset from today.
    
    Args:
        status (str, optional): Filter tasks by status. Must be one of: 'pending', 'in_progress', 'completed', 'cancelled' or None (no filtering).
    
    Returns:
        dict: A structured response containing next week's tasks:
            - status (str): Either 'success' or 'error'
            - tasks (list): List of task objects due next week (only on success)
            - count (int): Number of tasks returned (only on success)
            - filters (dict): Applied filters for reference (only on success)
            - message (str): Human-readable status message (only on error)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    next_week = date.today() + timedelta(days=7)
    return list_tasks(timeframe="week", status=status, target_date=next_week.isoformat(), include_stats=False)


def list_next_month_tasks(status: Optional[str]) -> Dict[str, Any]:
    """
    List tasks scheduled for next month based on their due dates.
    
    This function retrieves all tasks that are due during next month,
    using the due_date field as the source of truth rather than the stored timeframe field.
    The month calculation properly handles year transitions (December to January).
    
    Args:
        status (str, optional): Filter tasks by status. Must be one of: 'pending', 'in_progress', 'completed', 'cancelled' or None (no filtering).
    
    Returns:
        dict: A structured response containing next month's tasks:
            - status (str): Either 'success' or 'error'
            - tasks (list): List of task objects due next month (only on success)
            - count (int): Number of tasks returned (only on success)
            - filters (dict): Applied filters for reference (only on success)
            - message (str): Human-readable status message (only on error)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    today = date.today()
    # Calculate next month
    if today.month == 12:
        next_month = today.replace(year=today.year + 1, month=1)
    else:
        next_month = today.replace(month=today.month + 1)
    
    return list_tasks(timeframe="month", status=status, target_date=next_month.isoformat(), include_stats=False)


def list_this_week_tasks(status: Optional[str]) -> Dict[str, Any]:
    """
    List tasks scheduled for this week based on their due dates.
    
    This function retrieves all tasks that are due during the current week (Monday to Sunday),
    using the due_date field as the source of truth rather than the stored timeframe field.
    Tasks without due dates fall back to using their created_at timestamp for filtering.
    
    Args:
        status (str, optional): Filter tasks by status. Must be one of: 'pending', 'in_progress', 'completed', 'cancelled' or None (no filtering).
    
    Returns:
        dict: A structured response containing this week's tasks:
            - status (str): Either 'success' or 'error'
            - tasks (list): List of task objects due this week (only on success)
            - count (int): Number of tasks returned (only on success)
            - filters (dict): Applied filters for reference (only on success)
            - message (str): Human-readable status message (only on error)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    return list_tasks(timeframe="week", status=status, target_date=None, include_stats=False)


def list_today_tasks(status: Optional[str]) -> Dict[str, Any]:
    """
    List tasks scheduled for today based on their due dates.
    
    This function retrieves all tasks that are due today, using the due_date field
    as the source of truth rather than the stored timeframe field. Tasks without
    due dates fall back to using their created_at timestamp for filtering.
    
    Args:
        status (str, optional): Filter tasks by status. Must be one of: 'pending', 'in_progress', 'completed', 'cancelled' or None (no filtering).
    
    Returns:
        dict: A structured response containing today's tasks:
            - status (str): Either 'success' or 'error'
            - tasks (list): List of task objects due today (only on success)
            - count (int): Number of tasks returned (only on success)
            - filters (dict): Applied filters for reference (only on success)
            - message (str): Human-readable status message (only on error)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    return list_tasks(timeframe="day", status=status, target_date=None, include_stats=False)


def list_tasks_for_date(target_date: str, timeframe: str, status: Optional[str]) -> Dict[str, Any]:
    """
    List tasks for a specific date and timeframe context based on their due dates.
    
    This function retrieves tasks that fall within the specified timeframe period
    relative to the target date, using the due_date field as the source of truth
    rather than the stored timeframe field. For example, with timeframe='week',
    it returns tasks due during the week that contains the target date.
    
    Args:
        target_date (str): The target date in ISO format (YYYY-MM-DD). This is a required field that serves as the reference point for timeframe calculations.
        timeframe (str): The timeframe context for filtering. Must be one of: 'day', 'week', 'month', 'year'.
        status (str, optional): Filter tasks by status. Must be one of: 'pending', 'in_progress', 'completed', 'cancelled' or None (no filtering).
    
    Returns:
        dict: A structured response containing tasks for the specified date and timeframe:
            - status (str): Either 'success' or 'error'
            - tasks (list): List of task objects matching the criteria (only on success)
            - count (int): Number of tasks returned (only on success)
            - filters (dict): Applied filters for reference (only on success)
            - message (str): Human-readable status message (only on error)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    # Handle defaults inside the function
    if not timeframe:
        timeframe = "day"
    return list_tasks(timeframe=timeframe, status=status, target_date=target_date, include_stats=False)


def update_task(task_id: str, title: Optional[str], description: Optional[str],
                timeframe: Optional[str], priority: Optional[str], 
                status: Optional[str], due_date: Optional[str],
                tags: Optional[List[str]]) -> Dict[str, Any]:
    """
    Update an existing task with new information.
    
    This function modifies the properties of an existing task identified by its unique ID.
    Only specified fields will be changed, and the updated_at timestamp will be 
    automatically set to the current time.
    
    Args:
        task_id (str): The unique identifier of the task to update. This is a required field.
        title (str, optional): New task title or None to keep current.
        description (str, optional): New task description or None to keep current.
        timeframe (str, optional): New timeframe ('day', 'week', 'month', 'year') or None to keep current.
        priority (str, optional): New priority ('low', 'medium', 'high') or None to keep current.
        status (str, optional): New status ('pending', 'in_progress', 'completed', 'cancelled') or None to keep current.
        due_date (str, optional): New due date (ISO format YYYY-MM-DD) or None to keep current.
        tags (List[str], optional): New tags list or None to keep current.
    
    Returns:
        dict: A structured response containing the task update result:
            - status (str): Either 'success' or 'error'
            - message (str): Human-readable status message
            - task (dict): The updated task object with all current values (only on success)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    try:
        updates = {}
        
        # Build updates dictionary from non-None parameters
        if title is not None and title != "None":
            updates["title"] = title
        if description is not None and description != "None":
            updates["description"] = description
        if timeframe is not None and timeframe != "None":
            updates["timeframe"] = timeframe
        if priority is not None and priority != "None":
            updates["priority"] = priority
        if status is not None and status != "None":
            updates["status"] = status
        if due_date is not None and due_date != "None":
            updates["due_date"] = due_date
        if tags is not None:
            updates["tags"] = tags
        
        # Validate specific field values
        if "timeframe" in updates:
            valid_timeframes = ["day", "week", "month", "year"]
            if updates["timeframe"].lower() not in valid_timeframes:
                return {
                    "status": "error",
                    "message": f"Invalid timeframe. Must be one of: {valid_timeframes}",
                    "error_code": "INVALID_TIMEFRAME"
                }
        
        if "status" in updates:
            valid_statuses = ["pending", "in_progress", "completed", "cancelled"]
            if updates["status"].lower() not in valid_statuses:
                return {
                    "status": "error",
                    "message": f"Invalid status. Must be one of: {valid_statuses}",
                    "error_code": "INVALID_STATUS"
                }
        
        if "priority" in updates:
            valid_priorities = ["low", "medium", "high"]
            if updates["priority"].lower() not in valid_priorities:
                return {
                    "status": "error",
                    "message": f"Invalid priority. Must be one of: {valid_priorities}",
                    "error_code": "INVALID_PRIORITY"
                }
        
        return task_manager.update_task(task_id, updates)
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to update task: {str(e)}",
            "error_code": "INTERNAL_ERROR"
        }


def delete_task(task_id: str) -> Dict[str, Any]:
    """
    Delete a task from the system permanently.
    
    This function removes a task from the task management system using its unique identifier.
    Once deleted, the task cannot be recovered, so this operation should be used with caution.
    
    Args:
        task_id (str): The unique identifier of the task to delete. This is a required field.
    
    Returns:
        dict: A structured response containing the task deletion result:
            - status (str): Either 'success' or 'error'
            - message (str): Human-readable status message
            - deleted_task_id (str): The ID of the deleted task (only on success)
            - deleted_task (dict): The complete task object that was deleted (only on success)
            - error_code (str): Error code for programmatic handling (only on error)
    """
    try:
        return task_manager.delete_task(task_id)
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to delete task: {str(e)}",
            "error_code": "INTERNAL_ERROR"
        }


# Create the Hera task management agent
root_agent = Agent(
    name="hera",
    model="gemini-2.0-flash",
    description="Hera is a comprehensive task management agent that helps organize and track tasks across different timeframes.",
    instruction="""You are Hera, a task management assistant. Your primary job is to help users create and manage tasks with proper dates and timeframes.

## CRITICAL RULE FOR TASK CREATION:
ALWAYS use parse_date_expression FIRST when creating any task, regardless of how the user expresses the date.

### WORKFLOW FOR CREATING TASKS:
1. **ALWAYS CALL parse_date_expression FIRST** - Even if the user says "today", "tomorrow", or any date expression
2. Use the returned "date" as the due_date parameter
3. Use the returned "suggested_timeframe" as the timeframe parameter  
4. Generate a helpful description based on the title and context
5. Then call add_task with the parsed information

### DATE PARSING EXAMPLES:
- User says "tomorrow" → CALL parse_date_expression("tomorrow") → get ISO date → use in add_task
- User says "next Friday" → CALL parse_date_expression("next Friday") → get ISO date → use in add_task
- User says "in 3 days" → CALL parse_date_expression("in 3 days") → get ISO date → use in add_task
- User says "2024-12-25" → CALL parse_date_expression("2024-12-25") → get ISO date → use in add_task

### NEVER CREATE TASKS WITHOUT CALLING parse_date_expression FIRST!

## Your Other Capabilities:
- List and filter tasks by timeframe, status, or criteria
- View tasks for specific time periods
- Update existing tasks
- Delete tasks
- Provide task statistics

## Task Parameters:
- **Timeframes**: day, week, month, year
- **Statuses**: pending, in_progress, completed, cancelled  
- **Priorities**: low, medium, high
- **Description**: Generate based on title and context if not provided. Leave empty if no context.

## Available Tools:
- parse_date_expression: ALWAYS use this first for any date mentioned
- add_task: Create tasks (only after parsing dates)
- list_today_tasks, list_tomorrow_tasks: Quick task lists
- list_this_week_tasks, list_next_week_tasks, list_next_month_tasks: Period lists
- list_tasks_for_date: Tasks for specific dates
- update_task, delete_task: Task management

## Response Format:
Always respond with structured JSON data for A2A compatibility. Be proactive about task organization and deadline reminders.
""",
    tools=[
        parse_date_expression,
        add_task, 
        list_tasks, 
        list_today_tasks,
        list_tomorrow_tasks,
        list_this_week_tasks,
        list_next_week_tasks,
        list_next_month_tasks,
        list_tasks_for_date,
        update_task, 
        delete_task
    ]
)
