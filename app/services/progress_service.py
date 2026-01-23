"""Service for tracking segmentation progress using in-memory storage."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProgressStep:
    """Represents a single progress step."""
    name: str
    status: str  # "pending", "in-progress", "completed", "error"
    message: str = ""
    timestamp: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class SegmentationProgress:
    """Tracks progress of a segmentation task."""
    scene_id: str
    status: str  # "pending", "in-progress", "completed", "error"
    current_step: int = 0
    total_steps: int = 6  # Validación, Lectura, Procesamiento, Máscara, Cálculo, Creación de segmentos
    steps: list[ProgressStep] | None = None
    error_message: str = ""
    result: dict | None = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.steps is None:
            self.steps = [
                ProgressStep(name="Validación del archivo TIFF", status="pending"),
                ProgressStep(name="Lectura de datos geoespaciales", status="pending"),
                ProgressStep(name="Procesamiento con modelo de segmentación", status="pending"),
                ProgressStep(name="Generación de máscara predicha", status="pending"),
                ProgressStep(name="Cálculo de área por clase (m²)", status="pending"),
                ProgressStep(name="Creación de segmentos en base de datos", status="pending"),
            ]

    def to_dict(self) -> dict:
        return {
            "sceneId": self.scene_id,
            "status": self.status,
            "currentStep": self.current_step,
            "totalSteps": self.total_steps,
            "steps": [step.to_dict() for step in self.steps],
            "errorMessage": self.error_message,
            "result": self.result,
            "createdAt": self.created_at.isoformat() if self.created_at else "",
        }

    def is_expired(self, minutes: int = 60) -> bool:
        """Check if progress data is older than specified minutes."""
        return datetime.now(timezone.utc) - self.created_at > timedelta(minutes=minutes)


class ProgressService:
    """Service to track segmentation progress."""

    def __init__(self):
        self._progress: Dict[str, SegmentationProgress] = {}

    def initialize_progress(self, scene_id: str) -> None:
        """Initialize progress tracking for a scene."""
        self._progress[scene_id] = SegmentationProgress(scene_id=scene_id, status="in-progress")
        logger.info(f"Progress tracking initialized for scene {scene_id}")

    def get_progress(self, scene_id: str) -> SegmentationProgress | None:
        """Get current progress for a scene."""
        return self._progress.get(scene_id)

    def update_step(
        self,
        scene_id: str,
        step_index: int,
        status: str,
        message: str = "",
        error: str = "",
    ) -> None:
        """Update a specific step in the progress."""
        progress = self._progress.get(scene_id)
        if not progress:
            return

        if 0 <= step_index < len(progress.steps):
            step = progress.steps[step_index]
            step.status = status
            step.message = message
            step.error = error
            step.timestamp = datetime.now(timezone.utc).isoformat()
            progress.current_step = step_index

            if status == "in-progress":
                logger.info(f"[Scene {scene_id}] Step {step_index + 1}: {step.name} - IN PROGRESS: {message}")
            elif status == "completed":
                logger.info(f"[Scene {scene_id}] Step {step_index + 1}: {step.name} - COMPLETED")
            elif status == "error":
                logger.error(f"[Scene {scene_id}] Step {step_index + 1}: {step.name} - ERROR: {error}")

    def complete_progress(
        self,
        scene_id: str,
        result: dict | None = None,
    ) -> None:
        """Mark the entire segmentation as completed successfully."""
        progress = self._progress.get(scene_id)
        if progress:
            progress.status = "completed"
            progress.result = result
            progress.current_step = len(progress.steps) - 1
            # Mark last step as completed if not already
            if progress.steps[-1].status != "completed":
                progress.steps[-1].status = "completed"
                progress.steps[-1].timestamp = datetime.now(timezone.utc).isoformat()
            logger.info(f"Segmentation completed for scene {scene_id}")

    def error_progress(self, scene_id: str, error_message: str) -> None:
        """Mark the segmentation as failed."""
        progress = self._progress.get(scene_id)
        if progress:
            progress.status = "error"
            progress.error_message = error_message
            logger.error(f"Segmentation failed for scene {scene_id}: {error_message}")

    def cleanup_old_progress(self, minutes: int = 60) -> int:
        """Remove progress data older than specified minutes. Returns count of removed items."""
        initial_count = len(self._progress)
        self._progress = {
            scene_id: progress
            for scene_id, progress in self._progress.items()
            if not progress.is_expired(minutes)
        }
        removed_count = initial_count - len(self._progress)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old progress entries")
        return removed_count


# Global instance
_progress_service = ProgressService()


def get_progress_service() -> ProgressService:
    """Get the global progress service instance."""
    return _progress_service
