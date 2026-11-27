#!/usr/bin/env bash
# Make the Renku CLI (`rnk`, github.com/SwissDataScienceCenter/renku-cli)
# available in the current Renku session.
#
# Renku sessions only keep repos and data connectors, so the pinned release is
# downloaded (checksum-verified) into a session-local bin directory whenever it
# is missing. `rnk` is installed as a wrapper that defaults the Renku instance
# and the project context to this session's project, then execs the real binary
# (`rnk-<version>`). Already-exported RENKU_CLI_* variables take precedence.
# The path to `rnk` is printed on stdout; progress goes to stderr.
#
# Usage:
#   bash scripts/dev/renku/ensure_rnk.sh && rnk launcher list
#   RNK="$(bash scripts/dev/renku/ensure_rnk.sh)"     # explicit path for launchers
#
# Environment:
#   RNK_INSTALL_DIR   target directory (default: $VIRTUAL_ENV/bin if writable,
#                     which is on PATH in Renku sessions, else $HOME/.local/bin)

set -euo pipefail

RNK_VERSION="v0.7.2"
RENKU_URL_DEFAULT="https://renkulab.io"

if [[ -n "${RNK_INSTALL_DIR:-}" ]]; then
  INSTALL_DIR="$RNK_INSTALL_DIR"
elif [[ -n "${VIRTUAL_ENV:-}" && -w "${VIRTUAL_ENV}/bin" ]]; then
  INSTALL_DIR="$VIRTUAL_ENV/bin"
else
  INSTALL_DIR="$HOME/.local/bin"
fi
REAL_BIN="$INSTALL_DIR/rnk-$RNK_VERSION"
WRAPPER="$INSTALL_DIR/rnk"

case "$(uname -m)" in
  x86_64)
    TARGET="x86_64-unknown-linux-musl"
    SHA256="40fff923d4f212c55e56fc11fe3b12f84d67a2b6a4628e22659affe7d5180770"
    ;;
  aarch64 | arm64)
    TARGET="aarch64-unknown-linux-musl"
    SHA256="9f0f82e6238cbf3c7ad1b37295d2430b987f2ea4766edc1b517ea8994803f367"
    ;;
  *)
    echo "ensure_rnk: unsupported architecture $(uname -m)" >&2
    exit 1
    ;;
esac

if [[ ! -x "$REAL_BIN" ]]; then
  ASSET="rnk-${TARGET}-${RNK_VERSION}.tar.gz"
  URL="https://github.com/SwissDataScienceCenter/renku-cli/releases/download/${RNK_VERSION}/${ASSET}"

  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT

  echo "ensure_rnk: downloading $ASSET" >&2
  curl -fsSL --retry 3 -o "$WORK/$ASSET" "$URL"
  echo "$SHA256  $WORK/$ASSET" | sha256sum --check --quiet - >&2

  mkdir -p "$WORK/extract"
  tar -xzf "$WORK/$ASSET" -C "$WORK/extract"
  FOUND="$(find "$WORK/extract" -type f -name rnk | head -n 1)"
  if [[ -z "$FOUND" ]]; then
    echo "ensure_rnk: no 'rnk' binary in $ASSET" >&2
    exit 1
  fi

  mkdir -p "$INSTALL_DIR"
  install -m 0755 "$FOUND" "$REAL_BIN"
  echo "ensure_rnk: installed $("$REAL_BIN" --version) at $REAL_BIN" >&2
fi

cat > "$WRAPPER.tmp" <<EOF
#!/usr/bin/env bash
export RENKU_CLI_RENKU_URL="\${RENKU_CLI_RENKU_URL:-$RENKU_URL_DEFAULT}"
if [[ -z "\${RENKU_CLI_PROJECT_CONTEXT:-}" && -n "\${RENKU_PROJECT_PATH:-}" ]]; then
  export RENKU_CLI_PROJECT_CONTEXT="\$RENKU_PROJECT_PATH"
fi
exec "$REAL_BIN" "\$@"
EOF
chmod 0755 "$WRAPPER.tmp"
mv -f "$WRAPPER.tmp" "$WRAPPER"

case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) echo "ensure_rnk: $INSTALL_DIR is not on PATH; add it or call $WRAPPER directly" >&2 ;;
esac
echo "$WRAPPER"
