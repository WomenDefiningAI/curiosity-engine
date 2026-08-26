#!/bin/sh
set -eu

SOURCE=""
NO_SETUP=0
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source) SOURCE=${2:?--source requires a path}; shift 2 ;;
    --no-setup) NO_SETUP=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

PYTHON=${PYTHON:-python3}
"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "Curiosity Engine requires Python 3.11 or newer." >&2
  exit 1
}

CURIOSITY_HOME=${CURIOSITY_HOME:-"$HOME/.curiosity-engine"}
RUNTIME="$CURIOSITY_HOME/runtime"
if [ -n "$SOURCE" ]; then
  INSTALL_TARGET="${SOURCE}[slack]"
else
  INSTALL_TARGET="curiosity-engine[slack] @ https://github.com/WomenDefiningAI/curiosity-engine/archive/refs/heads/main.zip"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "family_home=$CURIOSITY_HOME"
  echo "runtime=$RUNTIME"
  echo "install_target=$INSTALL_TARGET"
  echo "next=curiosity setup"
  exit 0
fi

umask 077
mkdir -p "$CURIOSITY_HOME/private" "$CURIOSITY_HOME/workspace"
if [ ! -x "$RUNTIME/bin/python" ]; then
  "$PYTHON" -m venv "$RUNTIME"
fi
"$RUNTIME/bin/python" -m pip install --upgrade pip
"$RUNTIME/bin/python" -m pip install "$INSTALL_TARGET"
echo "Curiosity Engine installed without cloning a source repository."
if [ "$NO_SETUP" -eq 0 ]; then
  CURIOSITY_HOME="$CURIOSITY_HOME" "$RUNTIME/bin/curiosity" setup
fi
