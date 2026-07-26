"""Persistent + in-process cache for exact GenZ scalar results.

GenZ's ``prefill_moddeling`` / ``decode_moddeling`` are deterministic given
their inputs but relatively expensive.  This module caches their scalar
``Latency`` / ``Throughput`` outputs plus the two explicit pipeline-parallel
latencies: one scheduler batch's full ``TraversalLatency`` and the
``SaturatedServiceLatency`` available only when a caller actually models
multiple disjoint batches in flight.  It never changes, buckets, or
approximates a GenZ input.

Storage
-------
- An in-process memo removes repeats inside one worker.
- A sqlite cache shares exact results across CM's persistent spawn workers and
  across invocations on the same host.

The default sqlite file is deliberately node-local, under ``/tmp`` (namespaced
by uid, hostname, and checkout).  The repository commonly lives on NFS, where
sqlite locking is not safe for many concurrent writers.  ``CM_GENZ_CACHE``
remains the authoritative override; ``0`` / ``off`` disables disk caching.  An
explicit path detected on a remote filesystem fails safely to the exact
in-process memo.

Corruption is never repaired by deleting the old file.  A corrupt derived
cache and its sqlite sidecars are atomically renamed to unique ``.corrupt.*``
files before an empty cache is created.  Any unavailable disk cache degrades to
the in-process memo and therefore cannot change a cost-model result.

Correctness
-----------
The key is the sha256 of every cost-affecting kwarg (``model`` serialized via
its name/architecture, calibration ``ceff``/``meff``, ``bits``, ``system_name``,
parallelism, ...) plus ``_GENZ_FORMULA_VERSION``.  Non-cost kwargs (``debug``,
``model_profilling``) are excluded.  Bump ``_GENZ_FORMULA_VERSION`` if GenZ's
modeling formulas themselves change (the only input not otherwise captured by
the key).
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import socket
import sqlite3
import threading
import uuid
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

# Bump when GenZ's prefill/decode math or the cached scalar-output contract
# changes (calibration coefficients are already in the key).
_GENZ_FORMULA_VERSION = "v5-single-scheduler-batch-pp-traversal"

# kwargs that do NOT affect the numeric output -> excluded from the key.
_NONCOST_KWARGS = {"debug", "model_profilling"}

# WAL requires all clients to share reliable POSIX shared-memory/file-locking
# semantics.  These filesystems are conservatively treated as remote.
_REMOTE_FILESYSTEMS = {
    "9p",
    "afs",
    "ceph",
    "cifs",
    "fuse.sshfs",
    "gpfs",
    "lustre",
    "nfs",
    "nfs4",
    "smb3",
    "smbfs",
}

_MEMO: Dict[str, Dict[str, float]] = {}
_MEMO_LOCK = threading.Lock()

# Every process owns its sqlite connection.  ``spawn`` naturally starts with
# no connection; the pid guard also makes accidental use after ``fork`` safe.
_CONN: Optional[sqlite3.Connection] = None
_CONN_PATH: Optional[str] = None
_CONN_PID: Optional[int] = None
_CONN_FILE_ID: Optional[Tuple[int, int]] = None
_CONN_LOCK = threading.RLock()
_DISABLED_DISK = False
_DISABLED_PATH: Optional[str] = None

_STATS = {"memo_hit": 0, "disk_hit": 0, "miss": 0}
_STATS_LOCK = threading.Lock()


def _checkout_namespace() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return hashlib.sha256(str(repo_root).encode("utf-8")).hexdigest()[:12]


def _disk_path() -> Optional[str]:
    """Resolve the sqlite path, or ``None`` when disk caching is disabled."""
    value = os.environ.get("CM_GENZ_CACHE", "").strip()
    if value.lower() in ("0", "off", "false", "no"):
        return None
    if value:
        return os.path.abspath(os.path.expanduser(value))

    # Do not inherit TMPDIR here: on cluster hosts it is frequently redirected
    # to the same NFS checkout.  Hostname in the namespace also prevents an
    # accidentally shared /tmp mount from becoming a cross-host sqlite WAL.
    try:
        uid = str(os.getuid())
    except AttributeError:  # pragma: no cover - Windows compatibility
        uid = "user"
    host = hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12]
    directory = Path("/tmp") / f"rag-stack-genz-cache-{uid}" / host
    return str(directory / f"{_checkout_namespace()}.sqlite")


def _file_id(path: str) -> Optional[Tuple[int, int]]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return int(stat.st_dev), int(stat.st_ino)


def _filesystem_type(path: str) -> Optional[str]:
    """Return Linux mount fstype for *path* using the longest mount prefix."""
    try:
        target = os.path.realpath(path)
        if not os.path.exists(target):
            target = os.path.dirname(target) or os.curdir
        target = os.path.realpath(target)
        best: Tuple[int, str] | None = None
        with open("/proc/self/mountinfo", "r", encoding="utf-8") as stream:
            for line in stream:
                left, separator, right = line.partition(" - ")
                if not separator:
                    continue
                fields = left.split()
                fs_fields = right.split()
                if len(fields) < 5 or not fs_fields:
                    continue
                mountpoint = fields[4].replace("\\040", " ")
                try:
                    common = os.path.commonpath((target, mountpoint))
                except ValueError:
                    continue
                if common != mountpoint:
                    continue
                candidate = (len(mountpoint), fs_fields[0].lower())
                if best is None or candidate[0] > best[0]:
                    best = candidate
        return best[1] if best is not None else None
    except OSError:
        return None


def _journal_mode(path: str) -> str:
    fstype = _filesystem_type(path)
    return "DELETE" if fstype in _REMOTE_FILESYSTEMS else "WAL"


def _remote_cache_filesystem(path: str) -> Optional[str]:
    """Return the unsafe remote filesystem type, if one is detected."""
    fstype = _filesystem_type(path)
    return fstype if fstype in _REMOTE_FILESYSTEMS else None


@contextmanager
def _exclusive_file_lock(path: str) -> Iterator[None]:
    """Cross-process advisory lock; callers may fall back if unavailable."""
    directory = os.path.dirname(path) or os.curdir
    os.makedirs(directory, mode=0o700, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX fallback
            yield
        else:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _integrity_marker_path(path: str) -> str:
    host = hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12]
    return f"{path}.integrity.{host}.json"


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return "unknown"


def _integrity_marker_matches(path: str) -> bool:
    identity = _file_id(path)
    if identity is None:
        return False
    try:
        payload = json.loads(Path(_integrity_marker_path(path)).read_text(
            encoding="utf-8"
        ))
        return (
            set(payload) == {"boot_id", "file_id"}
            and payload.get("boot_id") == _boot_id()
            and tuple(payload.get("file_id", ())) == identity
        )
    except (OSError, ValueError, TypeError):
        return False


def _write_integrity_marker(path: str) -> None:
    identity = _file_id(path)
    if identity is None:
        return
    marker = _integrity_marker_path(path)
    temporary = f"{marker}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    payload = {
        "boot_id": _boot_id(),
        "file_id": list(identity),
    }
    try:
        Path(temporary).write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, marker)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _quick_check(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA quick_check").fetchall()
    if rows != [("ok",)]:
        detail = "; ".join(str(row[0]) for row in rows[:3])
        raise sqlite3.DatabaseError(f"sqlite quick_check failed: {detail}")


def _is_corruption_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(fragment in message for fragment in (
        "database disk image is malformed",
        "file is not a database",
        "not a database",
        "database schema is corrupt",
        "sqlite quick_check failed",
    ))


def _quarantine_corrupt_database(path: str) -> list[str]:
    """Atomically preserve a corrupt generation and all relevant sidecars."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    token = f"{timestamp}.{os.getpid()}.{uuid.uuid4().hex}"
    quarantined: list[str] = []
    sources = (
        path,
        f"{path}-wal",
        f"{path}-shm",
        _integrity_marker_path(path),
    )
    for source in sources:
        if not os.path.lexists(source):
            continue
        destination = f"{source}.corrupt.{token}"
        os.replace(source, destination)
        quarantined.append(destination)
    return quarantined


def _close_conn_locked() -> None:
    global _CONN, _CONN_PATH, _CONN_PID, _CONN_FILE_ID
    if _CONN is not None:
        try:
            _CONN.close()
        except sqlite3.Error:
            pass
    _CONN = None
    _CONN_PATH = None
    _CONN_PID = None
    _CONN_FILE_ID = None


def _configure_connection(connection: sqlite3.Connection, path: str) -> None:
    connection.execute("PRAGMA busy_timeout=30000")
    mode = _journal_mode(path)
    actual_mode = str(connection.execute(
        f"PRAGMA journal_mode={mode}"
    ).fetchone()[0]).upper()
    if actual_mode != mode:
        logger.debug(
            "genz_cache requested journal %s but sqlite selected %s for %s",
            mode,
            actual_mode,
            path,
        )
        if mode == "DELETE":
            raise sqlite3.OperationalError(
                f"refusing unsafe {actual_mode} journal on remote cache {path}"
            )
    synchronous = "FULL" if mode == "DELETE" else "NORMAL"
    connection.execute(f"PRAGMA synchronous={synchronous}")


def _initialize_database(path: str) -> sqlite3.Connection:
    """Validate/create the cache while serializing first-use across workers."""
    directory = os.path.dirname(path) or os.curdir
    os.makedirs(directory, mode=0o700, exist_ok=True)
    init_lock = f"{path}.init.lock"
    with _exclusive_file_lock(init_lock):
        connection: Optional[sqlite3.Connection] = None
        try:
            connection = sqlite3.connect(
                path,
                check_same_thread=False,
                timeout=30.0,
            )
            _configure_connection(connection, path)
            if not _integrity_marker_matches(path):
                _quick_check(connection)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS genz "
                "(k TEXT PRIMARY KEY, latency REAL, throughput REAL, "
                "traversal_latency REAL, saturated_service_latency REAL)"
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(genz)")
            }
            if "traversal_latency" not in columns:
                connection.execute(
                    "ALTER TABLE genz ADD COLUMN traversal_latency REAL"
                )
            if "saturated_service_latency" not in columns:
                connection.execute(
                    "ALTER TABLE genz ADD COLUMN saturated_service_latency REAL"
                )
            # This catches a structurally valid sqlite file with an incompatible
            # user-created table without treating it as corruption.
            connection.execute(
                "SELECT k, latency, throughput, traversal_latency, "
                "saturated_service_latency FROM genz LIMIT 0"
            )
            connection.commit()
            _write_integrity_marker(path)
            return connection
        except sqlite3.DatabaseError as error:
            if connection is not None:
                connection.close()
            if not _is_corruption_error(error):
                raise
            quarantined = _quarantine_corrupt_database(path)
            logger.warning(
                "genz_cache quarantined corrupt derived cache %s as %s",
                path,
                quarantined,
            )
            connection = sqlite3.connect(
                path,
                check_same_thread=False,
                timeout=30.0,
            )
            _configure_connection(connection, path)
            connection.execute(
                "CREATE TABLE genz "
                "(k TEXT PRIMARY KEY, latency REAL, throughput REAL, "
                "traversal_latency REAL, saturated_service_latency REAL)"
            )
            connection.commit()
            _quick_check(connection)
            _write_integrity_marker(path)
            return connection


def _get_conn_locked() -> Optional[sqlite3.Connection]:
    global _CONN, _CONN_PATH, _CONN_PID, _CONN_FILE_ID
    global _DISABLED_DISK, _DISABLED_PATH

    path = _disk_path()
    pid = os.getpid()
    if path is None:
        if _CONN is not None:
            _close_conn_locked()
        _DISABLED_DISK = True
        _DISABLED_PATH = None
        return None

    # Environment changes in a long-lived optimizer process are respected.
    if _DISABLED_PATH != path:
        _DISABLED_DISK = False
        _DISABLED_PATH = path

    connection_is_current = (
        _CONN is not None
        and _CONN_PATH == path
        and _CONN_PID == pid
        and _CONN_FILE_ID is not None
        and _CONN_FILE_ID == _file_id(path)
    )
    if connection_is_current:
        return _CONN
    if _CONN is not None:
        _close_conn_locked()
    if _DISABLED_DISK:
        return None

    # SQLite's rollback journal is less fragile than WAL on a remote mount,
    # but correctness still depends on locking guarantees that NFS/Lustre/GPFS
    # deployments frequently disable or weaken.  An explicit remote override
    # therefore fails safely to the exact process-local memo rather than
    # risking another corrupt shared cache.  The default path is node-local.
    remote_fstype = _remote_cache_filesystem(path)
    if remote_fstype is not None:
        logger.warning(
            "genz_cache: refusing shared sqlite on remote filesystem %s at %s; "
            "using exact in-process memo only",
            remote_fstype,
            path,
        )
        _DISABLED_DISK = True
        _DISABLED_PATH = path
        return None

    try:
        _CONN = _initialize_database(path)
        _CONN_PATH = path
        _CONN_PID = pid
        _CONN_FILE_ID = _file_id(path)
        logger.debug("genz_cache sqlite at %s", path)
    except Exception as error:  # noqa: BLE001 - exact memo-only fallback
        logger.warning(
            "genz_cache: disk disabled for %s (%s); using in-process memo only",
            path,
            error,
        )
        _close_conn_locked()
        _DISABLED_DISK = True
        _DISABLED_PATH = path
    return _CONN


def _get_conn() -> Optional[sqlite3.Connection]:
    """Return this process's validated sqlite connection, if available."""
    with _CONN_LOCK:
        return _get_conn_locked()


def _recover_after_corruption_locked(path: str, error: BaseException) -> None:
    """Quarantine only the exact failed inode; another process may win."""
    failed_identity = _CONN_FILE_ID
    _close_conn_locked()
    try:
        with _exclusive_file_lock(f"{path}.init.lock"):
            current_identity = _file_id(path)
            if current_identity is not None and current_identity == failed_identity:
                quarantined = _quarantine_corrupt_database(path)
                logger.warning(
                    "genz_cache quarantined corrupt derived cache %s after %s as %s",
                    path,
                    error,
                    quarantined,
                )
    except Exception as recovery_error:  # noqa: BLE001 - memo-only fallback
        logger.warning("genz_cache corruption recovery skipped: %s", recovery_error)


def _disk_lookup(key: str) -> Tuple[bool, Optional[Dict[str, float]]]:
    """Return ``(disk_available, value)``; absence is not an error."""
    with _CONN_LOCK:
        connection = _get_conn_locked()
        if connection is None:
            return False, None
        path = _CONN_PATH
        try:
            row = connection.execute(
                "SELECT latency, throughput, traversal_latency, "
                "saturated_service_latency FROM genz WHERE k=?", (key,)
            ).fetchone()
        except sqlite3.DatabaseError as error:
            if path is not None and _is_corruption_error(error):
                _recover_after_corruption_locked(path, error)
                connection = _get_conn_locked()
                if connection is not None:
                    row = connection.execute(
                        "SELECT latency, throughput, traversal_latency, "
                        "saturated_service_latency FROM genz WHERE k=?", (key,)
                    ).fetchone()
                else:
                    return False, None
            else:
                logger.debug("genz_cache read skipped: %s", error)
                return False, None
    if row is None:
        return True, None
    value = {"Latency": row[0], "Throughput": row[1]}
    if row[2] is not None:
        value["TraversalLatency"] = row[2]
    if row[3] is not None:
        value["SaturatedServiceLatency"] = row[3]
    return True, value


def _disk_store(key: str, value: Dict[str, float]) -> None:
    with _CONN_LOCK:
        connection = _get_conn_locked()
        if connection is None:
            return
        path = _CONN_PATH
        try:
            connection.execute(
                "INSERT OR IGNORE INTO genz "
                "(k, latency, throughput, traversal_latency, "
                "saturated_service_latency) VALUES (?,?,?,?,?)",
                (
                    key,
                    value["Latency"],
                    value["Throughput"],
                    value.get("TraversalLatency"),
                    value.get("SaturatedServiceLatency"),
                ),
            )
            connection.commit()
        except sqlite3.DatabaseError as error:
            if path is not None and _is_corruption_error(error):
                _recover_after_corruption_locked(path, error)
                connection = _get_conn_locked()
                if connection is not None:
                    connection.execute(
                        "INSERT OR IGNORE INTO genz "
                        "(k, latency, throughput, traversal_latency, "
                        "saturated_service_latency) VALUES (?,?,?,?,?)",
                        (
                            key,
                            value["Latency"],
                            value["Throughput"],
                            value.get("TraversalLatency"),
                            value.get("SaturatedServiceLatency"),
                        ),
                    )
                    connection.commit()
            else:
                logger.debug("genz_cache write skipped: %s", error)


@contextmanager
def _singleflight(path: str, key: str) -> Iterator[bool]:
    """Serialize the expensive first computation of one exact cache key."""
    lock_dir = f"{path}.key-locks"
    lock_path = os.path.join(lock_dir, key[:2], f"{key}.lock")
    lock = _exclusive_file_lock(lock_path)
    try:
        lock.__enter__()
    except OSError as error:
        # Disk caching is only an optimization.  If a filesystem does not
        # support advisory locks, compute normally rather than changing CM.
        logger.debug("genz_cache single-flight unavailable: %s", error)
        yield False
        return
    try:
        yield True
    finally:
        lock.__exit__(None, None, None)


def _model_key(model: Any) -> str:
    """Stable string for a GenZ model arg (str name or ModelConfig object)."""
    if isinstance(model, str):
        return model
    data = getattr(model, "__dict__", None)
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=True, default=str)
    return repr(model)


def _key(fn_name: str, kwargs: Dict[str, Any]) -> str:
    payload = {"fn": fn_name, "ver": _GENZ_FORMULA_VERSION}
    for name, value in kwargs.items():
        if name in _NONCOST_KWARGS:
            continue
        payload[name] = _model_key(value) if name == "model" else value
    # vLLM permits a structural layer-partition override.  GenZ resolves the
    # same environment variable when no explicit value is supplied, so it is
    # part of the exact cache identity (never a fitted topology coefficient).
    if kwargs.get("pipeline_layer_partition") is None:
        payload["vllm_pp_layer_partition_env"] = os.environ.get(
            "VLLM_PP_LAYER_PARTITION"
        )
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _increment_stat(name: str) -> None:
    with _STATS_LOCK:
        _STATS[name] += 1


def scheduler_batch_traversal_latency_ms(
    output: Dict[str, float],
    *,
    pipeline_parallel: int,
) -> float:
    """Return service for one non-overlapped vLLM scheduler batch.

    GenZ also publishes ``SaturatedServiceLatency=max(PP rank)``.  That value
    is a valid interdeparture only when a caller explicitly models enough
    *disjoint* scheduler batches in flight to fill the PP queue.  Current CM
    production consumers, including :class:`SimEngine`, own one in-flight
    scheduler step, so they must pay the complete traversal.  A PP=1 result is
    allowed to use the historical ``Latency`` scalar because traversal and
    saturated service are identical there.
    """

    if isinstance(pipeline_parallel, bool):
        raise ValueError("pipeline_parallel must be a positive integer")
    try:
        pp = int(pipeline_parallel)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("pipeline_parallel must be a positive integer") from exc
    if pp < 1 or pipeline_parallel != pp:
        raise ValueError("pipeline_parallel must be a positive integer")

    if pp > 1:
        value = output.get("TraversalLatency")
        if value is None:
            raise ValueError(
                "PP>1 single scheduler-batch pricing requires GenZ "
                "TraversalLatency; saturated max-rank service is not a "
                "single-batch latency"
            )
    else:
        value = output.get("TraversalLatency", output.get("Latency"))
    try:
        latency = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("GenZ traversal latency must be numeric") from exc
    if not (latency > 0.0) or not math.isfinite(latency):
        raise ValueError("GenZ traversal latency must be finite and positive")
    return latency


def _call(
    fn_name: str,
    real_fn: Callable[..., Any],
    kwargs: Dict[str, Any],
) -> Dict[str, float]:
    key = _key(fn_name, kwargs)

    with _STATS_LOCK:
        total = sum(_STATS.values())
        snapshot = dict(_STATS)
    if total and total % 5000 == 0:
        logger.info("genz_cache stats @%d calls: %s", total, snapshot)

    with _MEMO_LOCK:
        hit = _MEMO.get(key)
    if hit is not None:
        _increment_stat("memo_hit")
        return hit

    disk_available, hit = _disk_lookup(key)
    if hit is not None:
        with _MEMO_LOCK:
            _MEMO[key] = hit
        _increment_stat("disk_hit")
        return hit

    path = _disk_path()
    if disk_available and path is not None:
        with _singleflight(path, key):
            # A different process may have populated the row while this one
            # waited for the per-key lock.
            disk_available, hit = _disk_lookup(key)
            if hit is not None:
                with _MEMO_LOCK:
                    _MEMO[key] = hit
                _increment_stat("disk_hit")
                return hit

            _increment_stat("miss")
            output = real_fn(**kwargs)
            value = {
                "Latency": float(output["Latency"]),
                "Throughput": float(output.get("Throughput", 0.0)),
            }
            for field in ("TraversalLatency", "SaturatedServiceLatency"):
                if output.get(field) is not None:
                    value[field] = float(output[field])
            with _MEMO_LOCK:
                _MEMO[key] = value
            if disk_available:
                _disk_store(key, value)
            return value

    _increment_stat("miss")
    output = real_fn(**kwargs)
    value = {
        "Latency": float(output["Latency"]),
        "Throughput": float(output.get("Throughput", 0.0)),
    }
    for field in ("TraversalLatency", "SaturatedServiceLatency"):
        if output.get(field) is not None:
            value[field] = float(output[field])
    with _MEMO_LOCK:
        _MEMO[key] = value
    return value


def cached_prefill_moddeling(**kwargs: Any) -> Dict[str, float]:
    from GenZ.LLM_inference import prefill_moddeling as real

    return _call("prefill", real, kwargs)


def cached_decode_moddeling(**kwargs: Any) -> Dict[str, float]:
    from GenZ.LLM_inference import decode_moddeling as real

    return _call("decode", real, kwargs)


def cache_stats() -> Dict[str, int]:
    """Return ``{memo_hit, disk_hit, miss}`` since process start."""
    with _STATS_LOCK:
        return dict(_STATS)


def _reset_cache_state_for_tests() -> None:
    """Close/reset module state; intentionally private and test-only."""
    global _DISABLED_DISK, _DISABLED_PATH
    with _CONN_LOCK:
        _close_conn_locked()
        _DISABLED_DISK = False
        _DISABLED_PATH = None
    with _MEMO_LOCK:
        _MEMO.clear()
    with _STATS_LOCK:
        for name in _STATS:
            _STATS[name] = 0
