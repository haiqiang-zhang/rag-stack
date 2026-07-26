"""Errors raised by the isolated calibration artifact store."""


class CalibrationStoreError(RuntimeError):
    """Base class for calibration publication and snapshot failures."""


class CalibrationValidationError(CalibrationStoreError, ValueError):
    """A calibration document or manifest violated its contract."""


class CalibrationIntegrityError(CalibrationStoreError):
    """A content-addressed object did not match its declared digest."""


class IncompleteGenerationError(CalibrationValidationError):
    """A generation did not contain exactly its declared artifact set."""


class NoActiveCalibrationError(CalibrationStoreError, FileNotFoundError):
    """The store has no active complete calibration generation."""
