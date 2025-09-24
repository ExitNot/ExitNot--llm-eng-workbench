# Hera - Task Management Agent

TODO: task meneger have to be in toolContext???

Hera is a Google ADK (Agent Development Kit) agent designed to manage tasks across different timeframes (day, week, month, year) with structured JSON responses for A2A (Agent-to-Agent) communication.

## Features

- **Task Management**: Add, list, update, and delete tasks
- **Timeframe Support**: Organize tasks by day, week, month, or year
- **Status Tracking**: Track task progress (pending, in_progress, completed, cancelled)
- **Priority Levels**: Set task priorities (low, medium, high)
- **Local Storage**: Tasks stored locally in JSON format
- **Structured JSON Responses**: Compatible with A2A protocols

## Setup

1. **Install Dependencies**:
   ```bash
   pip install google-adk
   ```

2. **Environment Configuration**:
   Create a `.env` file with your Google API key:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

3. **Run the Agent**:
   ```bash
   adk web
   ```

## Available Tools

### add_task
Add a new task to the system.

**Parameters:**
- `title` (required): Task title
- `description`: Task description
- `timeframe`: Task timeframe (day, week, month, year) - default: "day"
- `priority`: Task priority (low, medium, high) - default: "medium"
- `due_date`: Due date in ISO format (optional)
- `tags`: List of tags (optional)

**Example Response:**
```json
{
  "status": "success",
  "message": "Task added successfully",
  "task": {
    "id": "uuid-string",
    "title": "Complete project",
    "description": "Finish the task management agent",
    "timeframe": "week",
    "priority": "high",
    "status": "pending",
    "created_at": "2025-09-23T10:00:00",
    "updated_at": "2025-09-23T10:00:00"
  }
}
```

### list_tasks
List tasks with optional filtering.

**Parameters:**
- `timeframe`: Filter by timeframe (optional)
- `status`: Filter by status (optional)
- `include_stats`: Include task statistics (optional)

**Example Response:**
```json
{
  "status": "success",
  "tasks": [...],
  "count": 5,
  "filters": {
    "timeframe": "week",
    "status": null
  }
}
```

### update_task
Update an existing task.

**Parameters:**
- `task_id` (required): Task ID to update
- `updates` (required): Dictionary of fields to update

**Example Response:**
```json
{
  "status": "success",
  "message": "Task updated successfully",
  "task": {...}
}
```

### delete_task
Delete a task by ID.

**Parameters:**
- `task_id` (required): Task ID to delete

**Example Response:**
```json
{
  "status": "success",
  "message": "Task deleted successfully",
  "deleted_task_id": "uuid-string"
}
```

## Project Structure

```
task-agent/
├── hera/
│   ├── __init__.py
│   ├── agent.py          # Main agent configuration
│   └── task_manager.py   # Task storage and management logic
├── main.py              # Original entry point (unused)
├── pyproject.toml       # Project dependencies
├── README.md           # This file
└── tasks.json          # Local task storage (created automatically)
```

## Usage Examples

Once the agent is running, you can interact with it through the web interface or programmatically:

1. **Add a daily task**:
   ```
   add_task("Review emails", "Check and respond to important emails", "day", "medium")
   ```

2. **List weekly tasks**:
   ```
   list_tasks("week")
   ```

3. **Update task status**:
   ```
   update_task("task-id", {"status": "in_progress"})
   ```

4. **Delete completed task**:
   ```
   delete_task("task-id")
   ```

## A2A Compatibility

All responses are structured JSON with consistent format:
- `status`: "success" or "error"
- `message`: Human-readable message
- `error_code`: Error code for failed operations
- Additional data fields as appropriate

This ensures seamless integration with Agent-to-Agent communication protocols.
