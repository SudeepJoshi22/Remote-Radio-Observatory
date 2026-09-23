#!/usr/bin/env bash
set -eu

SOURCE_DIR=${RRO_ARCHIVE_SOURCE_DIR:-/home/sudeep/fm_observations}
REMOTE=${RRO_ARCHIVE_REMOTE:-gdrive-rro}
REMOTE_PATH=${RRO_ARCHIVE_PATH:-remote-radio-observatory/npz}
RETENTION_HOURS=${RRO_ARCHIVE_RETENTION_HOURS:-720}
MIN_FREE_GB=${RRO_ARCHIVE_MIN_FREE_GB:-1}
DEST="${REMOTE}:${REMOTE_PATH}"

log() {
    printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

if [ ! -d "$SOURCE_DIR" ]; then
    log "archive source does not exist: $SOURCE_DIR"
    exit 1
fi
command -v rclone >/dev/null 2>&1 || {
    log "rclone is not installed"
    exit 1
}

# Only the recorder's final .npz names match. Its .npz.tmp files are not
# eligible, and the recorder publishes a final name only after fsync/rename.
log "copying completed NPZ files from $SOURCE_DIR to $DEST"
if ! rclone copy "$SOURCE_DIR" "$DEST" \
        --include '*.npz' \
        --exclude '*.tmp' \
        --stats 1m \
        --log-level INFO; then
    log "ERROR: NPZ upload failed; local recordings were left untouched"
    exit 1
fi

# This operates only inside the archive folder and never touches Pi files.
log "removing archive objects older than ${RETENTION_HOURS}h"
rclone delete "$DEST" \
    --include '*.npz' \
    --min-age "${RETENTION_HOURS}h" \
    --drive-use-trash=false \
    --log-level INFO

# 'about --json' is supported by Google Drive and gives an auditable quota
# check. A low quota is a warning/failure, not a reason to delete local data.
about_json=$(rclone about "$DEST" --json)
free_bytes=$(printf '%s' "$about_json" | python3 -c \
    'import json, sys; d=json.load(sys.stdin); v=d.get("free"); print(int(v) if v is not None else -1)')
if [ "$free_bytes" -lt 0 ]; then
    log "warning: rclone did not report Google Drive free quota"
    exit 1
fi
min_free_bytes=$((MIN_FREE_GB * 1000 * 1000 * 1000))
if [ "$free_bytes" -lt "$min_free_bytes" ]; then
    log "WARNING: Google Drive free quota is ${free_bytes} bytes (< ${MIN_FREE_GB} GB)"
    exit 1
fi
log "Google Drive free quota: ${free_bytes} bytes"
