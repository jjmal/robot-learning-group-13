#!/bin/bash

NAME="${1}"
TASK="${2}"
START="${3}"
END="${4}"
LOCAL_DIR="/ephemeral/data_raw"


if [[ -z "NAME" || -z "$TASK" || -z "$START" || -z "$END" ]]; then
    echo "Usage: $0 <name> <task> <start> <end>"
    echo "  e.g. $0 myorg 1 20 /data/datasets/"
    exit 1
fi

mkdir -p "$LOCAL_DIR"
 
echo "Downloading ${NAME}/t${TASK}-e{${START}..${END}} → ${LOCAL_DIR}"
echo "-----------------------------------------------------------"

FAILED=()

for x in $(seq "$START" "$END"); do
    REPO="${NAME}/t${TASK}-e${x}"
    DEST="${LOCAL_DIR}/t${TASK}-e${x}"
    echo "[${x}/${END}] Downloading ${REPO} ..."
 
    if hf download \
        --repo-type dataset \
        --local-dir "$DEST" \
        "$REPO"; then
        echo "  ✓ Done → ${DEST}"
    else
        echo "  ✗ Failed: ${REPO}"
        FAILED+=("$REPO")
    fi
 
    echo ""
done
 
echo "==========================================================="
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All datasets downloaded successfully."
else
    echo "Failed downloads (${#FAILED[@]}):"
    for repo in "${FAILED[@]}"; do
        echo "  - ${repo}"
    done
    exit 1
fi