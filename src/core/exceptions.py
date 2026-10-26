"""Standard application exception hierarchy with safe user-facing messaging."""


class DataAnalystError(Exception):
    """Base exception for all CelestiusAI application errors."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or "An unexpected system error occurred. Please try again."

    def __str__(self) -> str:
        return self.message


class ConfigurationError(DataAnalystError):
    """Raised when application configuration or environment settings are invalid."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(
            message,
            user_message or "Invalid application configuration. Please check your settings.",
        )


class DataIngestionError(DataAnalystError):
    """Raised when data ingestion, file reading, or parsing fails."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(
            message,
            user_message
            or "Failed to load the dataset. Please ensure the file format is supported.",
        )


class DataProfilingError(DataAnalystError):
    """Raised when data schema inference or quality profiling encounters an error."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(
            message,
            user_message
            or "Failed to profile the dataset structure. Please review column formats.",
        )


class AnalyticalExecutionError(DataAnalystError):
    """Raised when a statistical test, transformation, or analytical tool fails."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(
            message,
            user_message or "The requested analysis could not be completed with the current data.",
        )


class AgentOrchestrationError(DataAnalystError):
    """Raised when workflow planning, tool dispatching, or agent iteration fails."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(
            message,
            user_message or "The analysis planner encountered an error during investigation.",
        )


class EvidenceValidationError(DataAnalystError):
    """Raised when evidence verification fails or narrative claims contradict metrics."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(
            message,
            user_message or "Validation check failed: findings could not be reconciled with data.",
        )


class ReportGenerationError(DataAnalystError):
    """Raised when generating or exporting an executive report fails."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(
            message,
            user_message or "Failed to compile the intelligence report. Please try again.",
        )
