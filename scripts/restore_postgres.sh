#!/usr/bin/env bash
set -euo pipefail

backup_path=${1:-}
confirmation=${2:-}

if [[ -z "$backup_path" || ! -f "$backup_path/postgres.dump" || ! -f "$backup_path/uploads.tar.gz" ]]; then
  echo "usage: $0 PATH_TO_BACKUP_DIRECTORY RESTORE" >&2
  exit 2
fi
if [[ "$confirmation" != "RESTORE" ]]; then
  echo "restore replaces the current database; pass RESTORE to confirm" >&2
  exit 2
fi

docker compose --env-file .env.production -f docker-compose.production.yml \
  exec -T db pg_restore -U enclave -d enclave --clean --if-exists \
  --no-owner < "$backup_path/postgres.dump"
docker compose --env-file .env.production -f docker-compose.production.yml \
  run --rm --no-deps -T api sh -c \
  'find /app/data/uploads -mindepth 1 -delete && tar -C /app/data/uploads -xzf -' \
  < "$backup_path/uploads.tar.gz"

echo "restore complete: $backup_path"
