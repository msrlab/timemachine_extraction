Below is an updated README.md with concise explanations for each relevant command-line argument.

# Extract Selected Files From Time Machine Disks

For years, I had older Time Machine disks sitting on my shelf in a corrupt state, accessible only in read-only mode. I hesitated to wipe them clean, fearing I might have overlooked important files that could be needed someday.

This script consolidates multiple backups into a single location, allowing you to skip specific directories or file types and avoid duplicating identical files. In my case, I successfully merged two 4TB disks and one 2TB disk into a 400GB folder, which I can now archive with peace of mind.

## How it works

This Python script iterates through a Time Machine backup directory and copies only unique user-edited files to a specified output directory. Files are considered "unique" if their SHA-256 hash is distinct. This script was co-written with GPT o1 model.

> **Disclaimer**  
> This script worked for me personally while curating three older Time Machine volumes stored on my hard disk. However, there are no guarantees of completeness, correctness, or fitness for any particular purpose. Use it at your own risk, and always keep separate backups of your data!

## Features

- HFS+ (traditional Time Machine) and APFS support.
- Uses a JSON database to skip re-hashing files already processed.
- Optionally verifies file integrity after copying (to detect copy corruption).
- Skips system files and virtual machine disk images by default.
- Resume mode with optional skipping of large file hash checks if metadata matches previously extracted files.

## Requirements

- **Python 3.x**
- **macOS environment** (for Time Machine).
- **tmutil** (macOS command-line tool) if you want to enumerate APFS snapshots with `--tmutil-apfs`.

## Caveats and Potential Quirks

1. **Separate `metadata_db.json`**  
   It might be best to create a dedicated `metadata_db.json` for each Time Machine volume you process.  
2. **APFS Snapshots**  
   Requires `--tmutil-apfs`. Without it, APFS snapshots will not be enumerated.  
3. **Directory Skipping**  
   The script is opinionated on which directories or extensions to skip. Adjust as needed.  
4. **No Guarantees**  
   This is not an official tool. It worked for me, but use with caution.  
5. **Permission Issues**  
   You may need elevated privileges to read Time Machine snapshots.  
6. **Interruptions & Resuming**  
   The script saves its state (hash DB and metadata DB) on interrupt, allowing you to resume later.

## Usage

1. **Clone or download** the script (e.g., `extract_from_tm.py`) into a working directory.
2. **Create** or have ready an **output directory** where unique files will be placed.
3. **Run** the script from a terminal, supplying arguments as needed. Example:

```bash
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

For APFS Time Machine volumes:

python3 extract_from_tm.py \
    --backup-path /Volumes/Passport2TB \
    --output-path /my/extracted_files \
    --hash-db ./hash_db.json \
    --metadata-db ./metadata_db.json \
    --resume-skiphashcheck yes \
    --large-file-threshold 5242880 \
    --save-interval 5000 \
    --console-interval 20 \
    --tmutil-apfs yes

Key Arguments
	•	--backup-path
Path to the Time Machine volume. For HFS+ backups, this is the directory containing Backups.backupdb. For APFS volumes, pass the top-level volume (e.g., /Volumes/Passport2TB).
	•	--output-path
Directory where extracted unique files will be placed.
	•	--hash-db
Path to the JSON file storing the SHA-256 hashes of previously copied files. Used to avoid duplicates.
	•	--metadata-db
Path to a JSON file storing metadata about processed files (mtime, size, hash). Helps resume logic.
	•	--resume-skiphashcheck
If yes, the script attempts to skip hashing large files by matching path/size/mtime to what’s in hash_db.json. Defaults to yes.
	•	--large-file-threshold
Size threshold (in bytes). Files at or above this size can potentially skip hashing when --resume-skiphashcheck=yes. Default is 5242880 (5 MB).
	•	--save-interval
Number of processed files between saving the hash DB and metadata DB. Helps prevent losing progress.
	•	--console-interval
Number of processed files between console progress updates.
	•	--verify-after-copy
If yes, the script re-hashes each file after copying to ensure integrity. If the hash mismatches, the copy is removed.
	•	--tmutil-apfs
Enables use of tmutil listbackups to find APFS snapshots. Required for APFS volumes.
	•	--auto-rebuild-on-corrupt
If set, automatically rebuilds the hash DB from the output path if the hash_db.json is corrupt.
	•	--debug
If present, logs detailed debug statements to the console. Otherwise logs are shown at INFO level.
	•	--log-file
Specifies the debug-level log file (default: recovery.log).

For more details, run:

python3 extract_from_tm.py --help

Contributing

Contributions and suggestions are welcome! Feel free to open an issue or pull request on GitHub.

License

This project is released under the MIT License.
Please see the LICENSE file for details.

Notes
	•	The code follows general Python conventions, but you may want to run additional tests or reviews to ensure it meets your needs.
	•	If you still suspect essential files may be left in your old time machine backup, don't delete them.

