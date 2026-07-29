"""Safe, bounded tools used by the production solver."""

from .evidence import (
    CandidateEvidenceWriter,
    CandidateWriter,
    EvidenceWriter,
    EvidenceError,
    validate_answer,
    write_candidate,
)
from .files import (
    FileEntry,
    FileToolError,
    SearchMatch,
    TaskFiles,
    extract_archive,
    list_archive,
    list_files,
    read_file,
    resolve_task_path,
    search_files,
)
from .python_exec import (
    ExecutionResult,
    PythonExecutionError,
    PythonExecutor,
    run_python,
    shared_cpu_semaphore,
)
from .status import (
    BoardStatusProvider,
    DashboardStatusProvider,
    GameStatusProvider,
    ProblemStatusProvider,
    StatusProviderError,
    playable_board_for_phase,
)
from .web import EventWebSession, WebResponse, WebSessionPool, WebToolError

FileTools = TaskFiles
PythonTool = PythonExecutor
WebSession = EventWebSession

__all__ = [
    "CandidateEvidenceWriter",
    "CandidateWriter",
    "BoardStatusProvider",
    "DashboardStatusProvider",
    "EvidenceError",
    "EvidenceWriter",
    "EventWebSession",
    "ExecutionResult",
    "FileEntry",
    "FileToolError",
    "FileTools",
    "GameStatusProvider",
    "ProblemStatusProvider",
    "PythonExecutionError",
    "PythonExecutor",
    "PythonTool",
    "SearchMatch",
    "StatusProviderError",
    "TaskFiles",
    "WebResponse",
    "WebSessionPool",
    "WebSession",
    "WebToolError",
    "extract_archive",
    "list_archive",
    "list_files",
    "read_file",
    "resolve_task_path",
    "run_python",
    "search_files",
    "shared_cpu_semaphore",
    "validate_answer",
    "playable_board_for_phase",
    "write_candidate",
]
