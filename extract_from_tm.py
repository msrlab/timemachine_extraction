#!/usr/bin/env python3
"""
Script Name: extract_from_tm.py

Description:
    This script iterates through a Time Machine backup directory and copies only
    unique user-edited files to a specified output directory. Files are considered
    "unique" if their SHA-256 hash is distinct.

Usage Example:
    # HFS+ example (old style)
    python3 extract_from_tm.py \
        --backup-path /Volumes/TimeMachineDrive/Backups.backupdb \
        --output-path /path/to/extracted_user_files \
        --hash-db ./hash_db.json \
        --metadata-db ./metadata_db.json \
        --resume-skiphashcheck yes \
        --large-file-threshold 5242880 \
        --save-interval 100 \
        --console-interval 50 \
        --verify-after-copy yes

    # APFS example (new style)
    python3 extract_from_tm.py \
        --backup-path /Volumes/Passport2TB \
        --output-path ./WMB_script_test \
        --hash-db ./hash_db.json \
        --metadata-db ./metadata_db.json \
        --resume-skiphashcheck yes \
        --large-file-threshold 5242880 \
        --save-interval 5000 \
        --console-interval 20 \
        --tmutil-apfs yes
"""

import os
import sys
import shutil
import hashlib
import logging
import argparse
import json
import time
import signal
import subprocess
from pathlib import Path


def configure_logging(debug: bool, log_file: str):
    """
    Configure logging to file and console with specified log levels.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all logs at root level

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)


SKIP_DIRS = (
    "System", "Library", "private", "opt", "cores",
    "Network", "home", "usr", "sbin", "bin", "etc", "var",
    "Applications", "Developer", "Docker", "VirtualBox VMs",
    "Parallels", "VMware Fusion",
    ".Spotlight-V100", ".fseventsd", ".DocumentRevisions-V100",
    ".DS_Store",
    "Caches", "Logs", "Preferences", "Application Support",
    ".Trash", ".config", ".ssh",
    ".RecoverySets",
)

SKIP_EXTENSIONS = (
    ".log", ".tmp", ".cache", ".app", ".vmdk", ".vmx", ".vdi", ".qcow2", ".ovf",
    ".vagrant", ".parallels", ".pvm", ".hdd", ".swp", ".bak", ".dmp", ".pkg"
)


def atomic_save_json(data, path: Path, indent=2):
    """
    Write JSON to a temporary file, then atomically replace `path` with it.
    Prevents partial/corrupt writes if interrupted mid-write.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        tmp_path.replace(path)
    except Exception as e:
        logging.error(f"Failed atomic JSON dump to {path}: {e}")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def default_hash_db_structure() -> dict:
    """
    Return a default structure for our hash DB.
    """
    return {
        "version": 3,
        "hashes": {
            # "sha256": { "dest": "...", "size": 12345, "source_snapshot": "..." }
        }
    }


def load_hash_db(db_path: Path, output_path: Path, auto_rebuild: bool = False) -> dict:
    """
    Load a JSON-based hash DB that includes metadata for each hash.
    If corrupt, attempt to rebuild if auto_rebuild is True or prompt user.
    """
    logger = logging.getLogger()

    if not db_path.exists():
        logger.info(f"No existing hash DB found at {db_path}, starting empty.")
        return default_hash_db_structure()

    try:
        with db_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict) or "hashes" not in data:
                logger.warning("Hash DB missing 'hashes' key or is invalid. Starting empty.")
                return default_hash_db_structure()
            return data
    except Exception as e:
        logger.error(f"Failed to load hash DB from {db_path}: {e}")
        if auto_rebuild:
            logger.info("Auto-rebuild mode enabled. Rebuilding hash DB from output path...")
            new_data = rebuild_hash_db_from_extracted(output_path, db_path)
            save_hash_db(db_path, new_data)
            return new_data
        else:
            answer = input("Detected corrupt hash DB. Attempt to rebuild from output path [y/n]? ").lower().strip()
            if answer.startswith("y"):
                logger.info("Rebuilding hash DB from output path...")
                new_data = rebuild_hash_db_from_extracted(output_path, db_path)
                save_hash_db(db_path, new_data)
                return new_data
            else:
                logger.warning("User opted not to rebuild. Starting with empty DB.")
                return default_hash_db_structure()


def save_hash_db(db_path: Path, hash_db_data: dict):
    """
    Save the hash DB to JSON (atomic).
    """
    atomic_save_json(hash_db_data, db_path)


def rebuild_hash_db_from_extracted(extracted_path: Path, db_path: Path,
                                   partial_save_interval: int = 50) -> dict:
    """
    Enumerate every file in 'extracted_path', compute its hash,
    and build/update a DB structure with the discovered hashes.
    """
    logger = logging.getLogger()
    new_db = default_hash_db_structure()
    new_hashes = new_db["hashes"]

    if not extracted_path.exists() or not extracted_path.is_dir():
        logger.warning(f"Output path '{extracted_path}' not found or not a directory. Returning empty DB.")
        return new_db

    print("Counting files...", end='', flush=True)
    file_list = []
    file_count = 0
    start_count_time = time.time()

    for root, dirs, files in os.walk(extracted_path):
        for f in files:
            fp = Path(root) / f
            if fp.is_file():
                file_list.append(fp)
                file_count += 1
                if file_count % 100 == 0:
                    elapsed = time.time() - start_count_time
                    print(
                        f"\rCounting files: {file_count} files found so far. "
                        f"Elapsed time: {elapsed:.2f} seconds.",
                        end='',
                        flush=True
                    )

    total_files = len(file_list)
    elapsed_total = time.time() - start_count_time
    print(f"\rCounting files completed: {total_files} files found in {elapsed_total:.2f} seconds.")
    logger.info(f"Rebuilding hash DB by hashing {total_files} files in {extracted_path}...")

    if total_files == 0:
        return new_db

    start_time = time.time()
    count_new_hashes = 0

    for i, file_path in enumerate(file_list, start=1):
        try:
            file_size = file_path.stat().st_size
            file_hash = compute_hash(file_path)
            if not file_hash:
                continue

            if file_hash not in new_hashes:
                new_hashes[file_hash] = {
                    "dest": str(file_path),
                    "size": file_size,
                    "source_snapshot": "recovered"
                }
                count_new_hashes += 1

        except Exception as e:
            logger.debug(f"Skipped {file_path} due to error: {e}")

        if i % 50 == 0 or i == total_files:
            elapsed = time.time() - start_time
            per_file_time = elapsed / i if i > 0 else 0
            remaining = total_files - i
            eta_seconds = remaining * per_file_time
            progress_line = (
                f"\rRehashing progress: {i}/{total_files} files done "
                f"({(i / total_files) * 100:.1f}%), ETA: ~{eta_seconds / 60:.1f} min"
            )
            sys.stdout.write(progress_line.ljust(80))
            sys.stdout.flush()

        if i % partial_save_interval == 0:
            save_hash_db(db_path, new_db)

    sys.stdout.write("\n")
    sys.stdout.flush()
    logger.info(f"Finished rebuilding hash DB: found {count_new_hashes} unique file hashes.")
    save_hash_db(db_path, new_db)
    return new_db


def load_metadata_db(metadata_db_path: Path) -> dict:
    """
    Load the JSON-based metadata DB. Returns an empty dict if missing or invalid.
    """
    logger = logging.getLogger()
    if not metadata_db_path.exists():
        logger.info(f"No existing metadata DB found at {metadata_db_path}, starting empty.")
        return {}
    try:
        with metadata_db_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("Metadata DB is not a dict. Starting empty.")
                return {}
            return data
    except Exception as e:
        logger.error(f"Failed to load metadata DB from {metadata_db_path}: {e}")
        logger.warning("Starting empty metadata DB.")
        return {}


def save_metadata_db(metadata_db_path: Path, metadata_dict: dict):
    """
    Atomically save the metadata DB as JSON.
    """
    atomic_save_json(metadata_dict, metadata_db_path)


def human_readable_size(num_bytes: float) -> str:
    """
    Convert a byte count into a human-readable format (KB, MB, GB, etc.).
    """
    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes/1024:.1f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes/(1024**2):.1f} MB"
    else:
        return f"{num_bytes/(1024**3):.1f} GB"


def compute_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Compute the SHA-256 hash of a file. Returns empty string if read fails.
    """
    sha256 = hashlib.sha256()
    try:
        with file_path.open("rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                sha256.update(data)
    except (OSError, IOError) as e:
        logging.error(f"Failed to read file for hashing: {file_path}, error: {e}")
        return ""
    return sha256.hexdigest()


def should_skip(path: Path) -> bool:
    """
    Decide if a path should be skipped based on directory names, hidden files,
    or file extensions.
    """
    logger = logging.getLogger()
    if path.name.startswith('.'):
        logger.debug(f"Skipping {path} because it starts with '.'")
        return True

    for skip_dir in SKIP_DIRS:
        if skip_dir in path.parts:
            logger.debug(f"Skipping {path} due to skip_dir '{skip_dir}' in path.parts")
            return True

    if path.suffix.lower() in SKIP_EXTENSIONS:
        logger.debug(f"Skipping {path} due to extension '{path.suffix.lower()}' in SKIP_EXTENSIONS")
        return True

    return False


def find_snapshots_hfs(backup_path: Path) -> list[Path]:
    """
    Find all snapshot directories within the .backupdb path, ignoring .RecoverySets.
    """
    snapshot_dirs = []
    for machine_dir in backup_path.iterdir():
        if not machine_dir.is_dir():
            continue
        if machine_dir.name.startswith('.RecoverySets'):
            continue
        for snapshot_dir in machine_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue
            if snapshot_dir.name.startswith('.RecoverySets'):
                continue
            snapshot_dirs.append(snapshot_dir)
    return snapshot_dirs


def list_apfs_backups(volume_path: Path) -> list[Path]:
    """
    Use `tmutil listbackups <volume_path>` to get paths to APFS backup snapshots.
    Returns them as a list of Path objects.
    """
    logger = logging.getLogger()
    cmd = ["tmutil", "listbackups", str(volume_path)]
    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            logger.info(f"Found APFS backup snapshots via `tmutil listbackups`:")
            for ln in lines:
                logger.info(f"  {ln}")
        backup_paths = [Path(ln) for ln in lines]
        return backup_paths
    except subprocess.CalledProcessError as e:
        logger.error(f"`tmutil listbackups` failed with exit code {e.returncode}. stderr: {e.stderr}")
        return []
    except Exception as e:
        logger.error(f"Failed to list APFS backups: {e}")
        return []


def detect_time_machine_layout(volume_path: Path) -> str:
    """
    If we see a 'Backups.backupdb' directory, assume HFS. Otherwise, assume APFS.
    """
    if (volume_path / "Backups.backupdb").is_dir():
        return "hfs"
    return "apfs"


def count_snapshot_silently(snapshot_dir: Path) -> tuple[int, int]:
    """
    Return (file_count, total_size) for the given snapshot directory,
    applying 'should_skip' logic to skip unwanted files/folders.
    """
    file_count = 0
    total_size = 0
    if not snapshot_dir.is_dir():
        return (0, 0)

    # Revised to walk the snapshot_dir directly:
    for root, dirs, files in os.walk(snapshot_dir):
        root_path = Path(root)
        if should_skip(root_path):
            dirs[:] = []
            continue

        for filename in files:
            file_path = root_path / filename
            if should_skip(file_path):
                continue
            try:
                st_info = file_path.stat()
                file_count += 1
                total_size += st_info.st_size
            except Exception:
                pass
    return file_count, total_size


def quickcheck_file_hash_in_db(
    snapshot_dir: Path,
    file_path: Path,
    file_size: int,
    file_mtime: float,
    hash_db_data: dict,
    output_path: Path,
    time_tolerance: float = 1.0
) -> str:
    """
    Attempt a quick check: see if a file with the same subpath, size, and (approx)
    the same mtime is already recorded in hash_db_data. If yes, return the existing
    SHA-256 hash, else "".
    """
    logger = logging.getLogger()

    try:
        file_relative_path = file_path.relative_to(snapshot_dir.parent.parent)
    except ValueError:
        return ""

    would_be_dest = output_path / file_relative_path
    if not would_be_dest.is_file():
        return ""

    for f_hash, info_dict in hash_db_data["hashes"].items():
        existing_dest_str = info_dict.get("dest", "")
        if not existing_dest_str:
            continue
        if Path(existing_dest_str) == would_be_dest:
            if info_dict.get("size", -1) != file_size:
                continue
            try:
                extracted_stat = Path(existing_dest_str).stat()
                if abs(extracted_stat.st_mtime - file_mtime) > time_tolerance:
                    continue
            except (OSError, IOError):
                continue

            logger.debug(f"Quickcheck matched file: {would_be_dest}")
            return f_hash
    return ""


def process_file(
    snapshot_dir: Path,
    file_path: Path,
    file_size: int,
    file_mtime: float,
    metadata_dict: dict,
    hash_db_data: dict,
    output_path: Path,
    resume_skiphashcheck: bool,
    large_file_threshold: int,
    inode_cache: dict
) -> tuple[str, bool]:
    """
    Decide whether to re-hash the file or do a quick check based on path/size/mtime.
    Includes an in-memory inode shortcut. Returns (file_hash, did_hash):
      file_hash: the computed or retrieved hash ("" if error)
      did_hash: True if we performed a fresh compute_hash
    """
    did_hash = False
    try:
        st = file_path.stat()
        inode_key = (st.st_dev, st.st_ino)
    except Exception:
        return "", did_hash

    if inode_key in inode_cache:
        return inode_cache[inode_key], did_hash

    file_path_str = str(file_path.resolve())
    meta_entry = metadata_dict.get(file_path_str)

    if meta_entry:
        same_size = (meta_entry["size"] == file_size)
        same_mtime = (abs(meta_entry["mtime"] - file_mtime) < 1.0)
        if same_size and same_mtime:
            file_hash = meta_entry["hash"]
        else:
            file_hash = compute_hash(file_path)
            did_hash = True
    else:
        if file_size >= large_file_threshold and resume_skiphashcheck:
            file_hash = quickcheck_file_hash_in_db(
                snapshot_dir=snapshot_dir,
                file_path=file_path,
                file_size=file_size,
                file_mtime=file_mtime,
                hash_db_data=hash_db_data,
                output_path=output_path
            )
            if not file_hash:
                file_hash = compute_hash(file_path)
                did_hash = True
        else:
            file_hash = compute_hash(file_path)
            did_hash = True

    if file_hash:
        inode_cache[inode_key] = file_hash
    return file_hash, did_hash


def process_snapshot(
    snapshot_dir: Path,
    snapshot_index: int,
    total_snapshots: int,
    output_path: Path,
    hash_db_data: dict,
    hash_db_path: Path,
    metadata_dict: dict,
    metadata_db_path: Path,
    save_interval: int,
    console_interval: int,
    global_start_time: float,
    global_bytes_processed: list,
    global_files_processed: list,
    resume_skiphashcheck: bool,
    large_file_threshold: int,
    inode_cache: dict,
    verify_after_copy: bool
):
    """
    Process a single snapshot directory, comparing file hashes, copying new ones,
    updating hash_db_data & metadata_dict. Includes optional post-copy verification.
    """
    logger = logging.getLogger()

    file_count, total_size = count_snapshot_silently(snapshot_dir)
    if file_count == 0 or total_size == 0:
        logger.info(f"Snapshot '{snapshot_dir}' appears to have no files. Skipping.")
        return

    snapshot_name = snapshot_dir.name
    try:
        relative_snapshot_str = str(snapshot_dir.relative_to(snapshot_dir.parent.parent))
    except ValueError:
        relative_snapshot_str = snapshot_name

    logger.info(
        f"Processing snapshot {snapshot_index}/{total_snapshots} — '{relative_snapshot_str}'"
    )
    logger.info(
        f"Enumerating files to process: {file_count} files, {human_readable_size(total_size)} total"
    )

    processed_count = 0
    processed_bytes = 0
    copied_count = 0
    copied_bytes = 0

    hashed_count = 0
    hashed_bytes = 0

    snapshot_start_time = time.time()
    known_hashes = set(hash_db_data["hashes"].keys())

    dirty_hash_db = False
    dirty_metadata = False

    # Revised: directly walk the snapshot directory instead of iterating subdirs.
    for root, dirs, files in os.walk(snapshot_dir):
        root_path = Path(root)
        if should_skip(root_path):
            dirs[:] = []
            continue

        for filename in files:
            file_path = root_path / filename
            if should_skip(file_path):
                continue

            try:
                st_info = file_path.stat()
            except Exception:
                continue

            file_size = st_info.st_size
            file_mtime = st_info.st_mtime
            processed_count += 1
            processed_bytes += file_size
            global_files_processed[0] += 1

            file_hash, did_hash = process_file(
                snapshot_dir=snapshot_dir,
                file_path=file_path,
                file_size=file_size,
                file_mtime=file_mtime,
                metadata_dict=metadata_dict,
                hash_db_data=hash_db_data,
                output_path=output_path,
                resume_skiphashcheck=resume_skiphashcheck,
                large_file_threshold=large_file_threshold,
                inode_cache=inode_cache
            )
            if did_hash:
                hashed_count += 1
                hashed_bytes += file_size

            if file_hash:
                if file_hash not in known_hashes:
                    try:
                        try:
                            rel_path = file_path.relative_to(snapshot_dir.parent.parent)
                        except ValueError:
                            rel_path = file_path.relative_to(snapshot_dir)

                        dest_file = output_path / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)

                        shutil.copy2(file_path, dest_file)
                        copied_count += 1
                        copied_bytes += file_size

                        if verify_after_copy:
                            new_hash = compute_hash(dest_file)
                            if new_hash != file_hash:
                                logger.error(
                                    f"Post-copy verification failed for {dest_file}. Removing partial file."
                                )
                                try:
                                    dest_file.unlink()
                                except Exception as e:
                                    logger.error(f"Could not remove corrupted file {dest_file}: {e}")
                                continue

                        hash_db_data["hashes"][file_hash] = {
                            "dest": str(dest_file),
                            "size": file_size,
                            "source_snapshot": snapshot_name
                        }
                        known_hashes.add(file_hash)
                        dirty_hash_db = True

                    except Exception as e:
                        logger.error(f"Failed to copy {file_path} to {dest_file}: {e}")

                old_meta = metadata_dict.get(str(file_path.resolve()), {})
                new_meta = {
                    "mtime": file_mtime,
                    "size": file_size,
                    "hash": file_hash
                }
                if (
                    old_meta.get("mtime") != file_mtime
                    or old_meta.get("size") != file_size
                    or old_meta.get("hash") != file_hash
                ):
                    metadata_dict[str(file_path.resolve())] = new_meta
                    dirty_metadata = True

            if processed_count % console_interval == 0 or processed_count == file_count:
                fraction_done = processed_bytes / total_size if total_size else 1
                elapsed = time.time() - snapshot_start_time
                current_throughput = processed_bytes / elapsed if elapsed > 0 else 0
                time_left_snapshot = 0
                if current_throughput > 0 and fraction_done < 1:
                    time_left_snapshot = (total_size - processed_bytes) / current_throughput

                full_run_str = ""
                if snapshot_index > 1:
                    total_elapsed = time.time() - global_start_time
                    total_bytes_so_far = global_bytes_processed[0] + processed_bytes
                    global_throughput = (
                        total_bytes_so_far / total_elapsed
                        if total_elapsed > 0 else 0
                    )
                    snapshots_left = total_snapshots - snapshot_index
                    bytes_left_current = max(0, (total_size - processed_bytes))
                    future_bytes = (snapshots_left * total_size) + bytes_left_current
                    time_left_full = (future_bytes / global_throughput) if global_throughput > 0 else 0
                    full_run_str = f", full run: ~{time_left_full/60:.1f} min"

                console_line = (
                    f"\rProcessed {processed_count}/{file_count} files "
                    f"({human_readable_size(processed_bytes)} of {human_readable_size(total_size)}), "
                    f"Hashed: {hashed_count} files ({human_readable_size(hashed_bytes)}), "
                    f"Copied: {copied_count} files ({human_readable_size(copied_bytes)}). "
                    f"Time left (snapshot): ~{time_left_snapshot/60:.1f} min{full_run_str}"
                )
                sys.stdout.write(console_line.ljust(160))
                sys.stdout.flush()

            if processed_count % save_interval == 0:
                if dirty_hash_db:
                    save_hash_db(hash_db_path, hash_db_data)
                    dirty_hash_db = False
                if dirty_metadata:
                    save_metadata_db(metadata_db_path, metadata_dict)
                    dirty_metadata = False

    sys.stdout.write("\n")
    sys.stdout.flush()

    global_bytes_processed[0] += processed_bytes

    if dirty_hash_db:
        save_hash_db(hash_db_path, hash_db_data)
    if dirty_metadata:
        save_metadata_db(metadata_db_path, metadata_dict)

    logger.info(
        f"Finished processing snapshot '{relative_snapshot_str}' — "
        f"copied {copied_count} new files ({human_readable_size(copied_bytes)})"
    )


def main():
    """
    Main function to parse arguments and orchestrate the file extraction
    from Time Machine backups.
    """
    parser = argparse.ArgumentParser(
        description="Extract unique user-edited files from a Time Machine backup."
    )
    parser.add_argument(
        "--hash-db",
        default="hash_db.json",
        help="Path to the JSON file where previously copied file hashes are stored."
    )
    parser.add_argument(
        "--metadata-db",
        default="metadata_db.json",
        help="Path to the JSON file storing file path->(mtime, size, hash)."
    )
    parser.add_argument(
        "--backup-path",
        required=True,
        help=(
            "Path to the Time Machine volume. For HFS+ backups, this is the directory "
            "containing Backups.backupdb. For APFS, pass the top-level volume (e.g. /Volumes/Passport2TB)."
        )
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to the folder where unique files will be copied."
    )
    parser.add_argument(
        "--log-file",
        default="recovery.log",
        help="Detailed log file (DEBUG-level) for diagnostic info."
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=2000,
        help="Number of processed files between DB saves."
    )
    parser.add_argument(
        "--console-interval",
        type=int,
        default=20,
        help="Number of processed files between console progress updates."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to console for troubleshooting."
    )
    parser.add_argument(
        "--auto-rebuild-on-corrupt",
        action="store_true",
        help="Automatically rebuild hash_db.json if it is corrupted, without prompting."
    )
    parser.add_argument(
        "--resume-skiphashcheck",
        choices=["yes", "no"],
        default="yes",
        help=(
            "If 'yes', when a file is missing from metadata_db, try matching its path/size/mtime "
            "in the hash_db to skip re-hashing if presumably from the same snapshot."
        )
    )
    parser.add_argument(
        "--large-file-threshold",
        type=int,
        default=5 * 1024 * 1024,  # 5 MB by default
        help=(
            "Size threshold (in bytes). Files >= this size may skip hashing if --resume-skiphashcheck=yes "
            "and a matching file is found in hash_db. Default is 5 MB."
        )
    )
    parser.add_argument(
        "--verify-after-copy",
        choices=["yes", "no"],
        default="no",
        help=(
            "If 'yes', after copying a file, re-hash it to ensure integrity. If mismatch, the copy is removed."
        )
    )
    parser.add_argument(
        "--tmutil-apfs",
        action="store_true",
        help="If set, for APFS volumes we use `tmutil listbackups` to find .backup snapshots."
    )

    args = parser.parse_args()
    configure_logging(args.debug, args.log_file)
    logger = logging.getLogger()

    def handle_interrupt(signum, frame):
        logger.warning("Received KeyboardInterrupt. Attempting final DB save, then exiting.")
        if 'hash_db_data' in globals():
            save_hash_db(Path(args.hash_db), hash_db_data)
        if 'metadata_dict' in globals():
            save_metadata_db(Path(args.metadata_db), metadata_dict)
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_interrupt)

    backup_path = Path(args.backup_path)
    output_path = Path(args.output_path)
    hash_db_path = Path(args.hash_db)
    metadata_db_path = Path(args.metadata_db)

    if not backup_path.exists():
        logger.error(f"Backup path does not exist: {backup_path}")
        sys.exit(1)

    logger.info("Reading hash DB...")
    hash_db_data = load_hash_db(
        db_path=hash_db_path,
        output_path=output_path,
        auto_rebuild=args.auto_rebuild_on_corrupt
    )

    logger.info("Reading metadata DB...")
    metadata_dict = load_metadata_db(metadata_db_path)

    output_path.mkdir(parents=True, exist_ok=True)

    layout = detect_time_machine_layout(backup_path)
    logger.info(f"Detected layout: {layout}")

    snapshot_dirs = []
    if layout == "hfs":
        snapshot_dirs = find_snapshots_hfs(backup_path)
    else:
        if args.tmutil_apfs:
            logger.info("Using tmutil listbackups on APFS volume...")
            snapshot_dirs = list_apfs_backups(backup_path)
        else:
            logger.error(
                "APFS volume detected, but --tmutil-apfs not specified. "
                "Cannot list or process APFS backups. Exiting."
            )
            sys.exit(1)

    total_snapshots = len(snapshot_dirs)
    if total_snapshots == 0:
        logger.info("No valid snapshots found. Exiting.")
        return

    logger.info(f"Found {total_snapshots} snapshot folders to process.\n")
    logger.info("Preparing to process the snapshots...")

    global_bytes_processed = [0]
    global_files_processed = [0]
    global_start_time = time.time()

    inode_cache = {}
    verify_after_copy_bool = (args.verify_after_copy == "yes")
    resume_skiphashcheck_bool = (args.resume_skiphashcheck == "yes")

    for i, sdir in enumerate(snapshot_dirs, start=1):
        process_snapshot(
            snapshot_dir=sdir,
            snapshot_index=i,
            total_snapshots=total_snapshots,
            output_path=output_path,
            hash_db_data=hash_db_data,
            hash_db_path=hash_db_path,
            metadata_dict=metadata_dict,
            metadata_db_path=metadata_db_path,
            save_interval=args.save_interval,
            console_interval=args.console_interval,
            global_start_time=global_start_time,
            global_bytes_processed=global_bytes_processed,
            global_files_processed=global_files_processed,
            resume_skiphashcheck=resume_skiphashcheck_bool,
            large_file_threshold=args.large_file_threshold,
            inode_cache=inode_cache,
            verify_after_copy=verify_after_copy_bool
        )

    logger.info("\nAll snapshots processed. Exiting.\n")


if __name__ == "__main__":
    main()