#!/usr/bin/env bash
#
# Upload CoDiT dataset archives to a Zenodo deposition via the bucket API.
#
# Usage:
#   export ZENODO_TOKEN=<token from https://zenodo.org/account/settings/applications/tokens/new>
#   scripts/upload_to_zenodo.sh                     # production Zenodo
#   ZENODO_SANDBOX=1 scripts/upload_to_zenodo.sh    # sandbox (for a dry run)
#   ZENODO_DEP_ID=1234567 scripts/upload_to_zenodo.sh  # resume into an existing deposition
#
# On success, prints the deposition ID, edit URL, and (after you add metadata
# and publish in the web UI) the record ID to paste into download_data.py.
#
# The script never publishes — you review metadata in the UI first.

set -euo pipefail

ARCHIVES_DIR="${ARCHIVES_DIR:-/mydata/adopt/christian/codit_archives}"

if [[ -z "${ZENODO_TOKEN:-}" ]]; then
    echo "ERROR: ZENODO_TOKEN is not set." >&2
    echo "Get a token at https://zenodo.org/account/settings/applications/tokens/new" >&2
    echo "and export it: export ZENODO_TOKEN=..." >&2
    exit 1
fi

if [[ ! -d "$ARCHIVES_DIR" ]]; then
    echo "ERROR: ARCHIVES_DIR=$ARCHIVES_DIR does not exist." >&2
    exit 1
fi

if [[ "${ZENODO_SANDBOX:-0}" == "1" ]]; then
    API="https://sandbox.zenodo.org/api"
    WEB="https://sandbox.zenodo.org"
    echo "Using SANDBOX Zenodo."
else
    API="https://zenodo.org/api"
    WEB="https://zenodo.org"
fi

# 1. Create a new deposition (or use the one supplied).
if [[ -n "${ZENODO_DEP_ID:-}" ]]; then
    DEP_ID="$ZENODO_DEP_ID"
    echo "Resuming upload into existing deposition $DEP_ID."
else
    echo "Creating new empty deposition..."
    CREATE_RESP=$(curl -fsS -X POST "$API/deposit/depositions" \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{}')
    DEP_ID=$(echo "$CREATE_RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
    echo "Created deposition: $DEP_ID"
fi

# 2. Get the bucket URL for this deposition.
DEP_JSON=$(curl -fsS "$API/deposit/depositions/$DEP_ID" \
    -H "Authorization: Bearer $ZENODO_TOKEN")
BUCKET=$(echo "$DEP_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["links"]["bucket"])')
echo "Bucket URL: $BUCKET"

# 3. Upload every archive in ARCHIVES_DIR.
shopt -s nullglob
ARCHIVES=("$ARCHIVES_DIR"/*.tar.gz)
if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
    echo "ERROR: no .tar.gz files under $ARCHIVES_DIR" >&2
    exit 1
fi

echo
echo "Uploading ${#ARCHIVES[@]} archives..."
for f in "${ARCHIVES[@]}"; do
    name=$(basename "$f")
    size=$(du -h "$f" | cut -f1)
    echo
    echo "-> $name ($size)"
    curl --fail --progress-bar -o /tmp/zenodo_upload.$$.json \
        -H "Authorization: Bearer $ZENODO_TOKEN" \
        --upload-file "$f" \
        "$BUCKET/$name"
    # Verify Zenodo's returned size matches local size.
    remote_size=$(python3 -c 'import json; print(json.load(open("/tmp/zenodo_upload.'$$'.json"))["size"])' 2>/dev/null || echo "?")
    local_size=$(stat -c %s "$f")
    if [[ "$remote_size" != "$local_size" ]]; then
        echo "  WARNING: size mismatch (local=$local_size remote=$remote_size)" >&2
    else
        echo "  OK ($local_size bytes verified)"
    fi
    rm -f /tmp/zenodo_upload.$$.json
done

echo
echo "======================================================================"
echo "All uploads complete."
echo "Deposition ID:  $DEP_ID"
echo "Edit URL:       $WEB/deposit/$DEP_ID"
echo
echo "Next steps:"
echo "  1. Open the Edit URL, add metadata (title, authors, description,"
echo "     license CC-BY 4.0, type Dataset)."
echo "  2. Click Publish."
echo "  3. Note the record ID from the resulting URL (zenodo.org/record/XXXXX)"
echo "     and update DATASET_URLS in scripts/download_data.py."
echo "======================================================================"
