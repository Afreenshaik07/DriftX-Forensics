from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import shutil
import json


@dataclass
class RegistryRecord:
    """
    Record of a champion model promotion or rollback.
    """

    timestamp: str
    action: str
    source_model: str
    champion_model: str
    reason: str

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelRegistry:
    """
    Lightweight local model registry for DriftX-Forensics.

    Responsibilities:
        - Maintain the current champion model.
        - Backup the previous champion.
        - Promote a validated challenger.
        - Roll back to the previous champion.
        - Record model lifecycle events.
    """

    def __init__(
        self,
        registry_dir: Path = Path("artifacts/registry"),
    ):
        self.registry_dir = Path(registry_dir)

        self.registry_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.champion_path = (
            self.registry_dir
            / "champion_rul_model.joblib"
        )

        self.backup_path = (
            self.registry_dir
            / "previous_champion_rul_model.joblib"
        )

        self.history_path = (
            self.registry_dir
            / "promotion_history.json"
        )

    def _timestamp(self) -> str:
        """
        Generate a UTC timestamp for registry events.
        """

        return datetime.utcnow().isoformat(
            timespec="seconds"
        ) + "Z"

    def _write_history(
        self,
        record: RegistryRecord,
    ) -> None:
        """
        Append a registry event to the history file.
        """

        history = []

        if self.history_path.exists():

            try:
                with open(
                    self.history_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    history = json.load(file)

            except (json.JSONDecodeError, OSError):
                history = []

        history.append(
            record.to_dict()
        )

        with open(
            self.history_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history,
                file,
                indent=4,
            )

    def initialize(
        self,
        model_path: Path,
        reason: str = "Initial champion registration",
    ) -> Dict:
        """
        Register the initial champion model.

        Existing champion is never overwritten silently.
        """

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        if self.champion_path.exists():
            return {
                "status": "EXISTS",
                "champion_path": str(
                    self.champion_path
                ),
                "message": (
                    "Champion already exists. "
                    "Initialization skipped."
                ),
            }

        shutil.copy2(
            model_path,
            self.champion_path,
        )

        record = RegistryRecord(
            timestamp=self._timestamp(),
            action="INITIALIZE",
            source_model=str(model_path),
            champion_model=str(
                self.champion_path
            ),
            reason=reason,
        )

        self._write_history(
            record
        )

        return {
            "status": "INITIALIZED",
            "champion_path": str(
                self.champion_path
            ),
            "record": record.to_dict(),
        }

    def promote(
        self,
        challenger_path: Path,
        reason: str = "Challenger passed safety gate",
    ) -> Dict:
        """
        Promote a challenger to champion.

        The existing champion is backed up before promotion.
        """

        challenger_path = Path(
            challenger_path
        )

        if not challenger_path.exists():
            raise FileNotFoundError(
                "Challenger model not found: "
                f"{challenger_path}"
            )

        # -----------------------------------------------------
        # Backup current champion
        # -----------------------------------------------------

        if self.champion_path.exists():

            shutil.copy2(
                self.champion_path,
                self.backup_path,
            )

        # -----------------------------------------------------
        # Promote challenger
        # -----------------------------------------------------

        shutil.copy2(
            challenger_path,
            self.champion_path,
        )

        record = RegistryRecord(
            timestamp=self._timestamp(),
            action="PROMOTE",
            source_model=str(
                challenger_path
            ),
            champion_model=str(
                self.champion_path
            ),
            reason=reason,
        )

        self._write_history(
            record
        )

        return {
            "status": "PROMOTED",
            "champion_path": str(
                self.champion_path
            ),
            "backup_path": str(
                self.backup_path
            ),
            "record": record.to_dict(),
        }

    def rollback(
        self,
        reason: str = "Safety rollback",
    ) -> Dict:
        """
        Restore the previous champion model.
        """

        if not self.backup_path.exists():
            return {
                "status": "FAILED",
                "message": (
                    "No previous champion backup "
                    "is available."
                ),
            }

        if self.champion_path.exists():

            rollback_backup = (
                self.registry_dir
                / "rejected_champion_rul_model.joblib"
            )

            shutil.copy2(
                self.champion_path,
                rollback_backup,
            )

        shutil.copy2(
            self.backup_path,
            self.champion_path,
        )

        record = RegistryRecord(
            timestamp=self._timestamp(),
            action="ROLLBACK",
            source_model=str(
                self.backup_path
            ),
            champion_model=str(
                self.champion_path
            ),
            reason=reason,
        )

        self._write_history(
            record
        )

        return {
            "status": "ROLLED_BACK",
            "champion_path": str(
                self.champion_path
            ),
            "record": record.to_dict(),
        }

    def get_champion_path(self) -> Optional[Path]:
        """
        Return the current champion path.
        """

        if not self.champion_path.exists():
            return None

        return self.champion_path

    def get_history(self):
        """
        Return model promotion history.
        """

        if not self.history_path.exists():
            return []

        try:
            with open(
                self.history_path,
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

        except (json.JSONDecodeError, OSError):
            return []