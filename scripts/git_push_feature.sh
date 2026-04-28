#!/usr/bin/env bash
# Push local main to a feature branch on GitHub over SSH (avoids broken HTTPS PAT / 403).
# One-time: add the printed public key at https://github.com/settings/keys
# You must have write access to the remote repo (or push to your fork and open a PR).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BRANCH="${1:-har-imu-labeled-stream}"
REMOTE_URL_SSH="git@github.com:aman31kmr/metawear-baro-tools.git"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh" || true
KEY="$HOME/.ssh/id_ed25519_metawear_github"
if [[ ! -f "$KEY" ]]; then
  echo "Creating SSH key: $KEY"
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "metawear-baro-tools-push-$(hostname -s 2>/dev/null || echo host)"
fi

if ! grep -q '^github.com' "$HOME/.ssh/known_hosts" 2>/dev/null; then
  echo "Adding github.com to known_hosts..."
  ssh-keyscan -t ed25519 github.com >>"$HOME/.ssh/known_hosts"
  chmod 644 "$HOME/.ssh/known_hosts"
fi

echo ""
echo "=== Add this SSH public key to GitHub (account that can push) ==="
echo "    https://github.com/settings/keys  -> New SSH key"
echo ""
cat "${KEY}.pub"
echo ""
echo "================================================================"
echo ""

git remote set-url origin "$REMOTE_URL_SSH"
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

echo "Pushing main -> origin/$BRANCH ..."
if git push -u origin "main:$BRANCH"; then
  echo "OK: https://github.com/aman31kmr/metawear-baro-tools/tree/$BRANCH"
else
  echo "Push failed. If you see 'Permission denied (publickey)', add the key above to GitHub."
  echo "If you are not a collaborator on aman31kmr/metawear-baro-tools, fork it and run:"
  echo "  git remote set-url origin git@github.com:YOUR_USER/metawear-baro-tools.git"
  echo "  $0 $BRANCH"
  exit 1
fi
