#!/usr/bin/env bash
set -euo pipefail

umask 077
backup_dir=${1:-./backups}
mkdir -p "$backup_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
target="$backup_dir/enclave-$stamp"
mkdir "$target"

docker compose --env-file .env.production -f docker-compose.production.yml \
  exec -T db pg_dump -U enclave -d enclave --format=custom --no-owner \
  > "$target/postgres.dump"
docker compose --env-file .env.production -f docker-compose.production.yml \
  run --rm --no-deps -T api tar -C /app/data/uploads -czf - . \
  > "$target/uploads.tar.gz"

test -s "$target/postgres.dump"
test -s "$target/uploads.tar.gz"
echo "backup created: $target"
