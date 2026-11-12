#!/usr/bin/env bash
#
# pgdb.sh — Dump & Restore der pgvector-Datenbank (ask-pdf)
#
# Nutzung:
#   ./pgdb.sh dump  [DATEI]       Dump erstellen (default: askpdf_YYYY-MM-DD_HHMMSS.dump)
#   ./pgdb.sh restore DATEI       Dump wiederherstellen
#
# Läuft pg_dump/pg_restore im Docker-Container "postgres".

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DB_USER="askpdf"
DB_NAME="askpdf"
CONTAINER="$(docker compose -f "$PROJECT_DIR/compose.yaml" ps -q postgres 2>/dev/null || true)"

# --- Hilfsfunktionen ----------------------------------------------------
usage() {
    sed -n '3,8p' "$0" | sed 's/^# \?//'
    exit 1
}

die() { echo "FEHLER: $*" >&2; exit 1; }

ensure_container() {
    [[ -n "$CONTAINER" ]] || die "Postgres-Container läuft nicht. Starte ihn mit: docker compose up -d postgres"
}

# --- Dump ---------------------------------------------------------------
do_dump() {
    ensure_container
    local outfile="${1:-askpdf_$(date +%Y-%m-%d_%H%M%S).dump}"

    echo "Erstelle Dump → $outfile"
    docker exec "$CONTAINER" \
        pg_dump -U "$DB_USER" -d "$DB_NAME" \
            --format=custom \
            --no-owner \
            --no-privileges \
        > "$outfile"

    echo "Fertig. $(du -h "$outfile" | cut -f1) geschrieben."
}

# --- Restore ------------------------------------------------------------
do_restore() {
    ensure_container
    local infile="${1:-}"
    [[ -n "$infile" ]] || die "Bitte Dump-Datei angeben.\n  ./pgdb.sh restore DATEI"
    [[ -f "$infile" ]] || die "Datei nicht gefunden: $infile"

    echo "ACHTUNG: Alle bestehenden Daten in 'documents' werden ersetzt!"
    read -rp "Fortfahren? [j/N] " confirm
    [[ "$confirm" =~ ^[jJyY]$ ]] || { echo "Abgebrochen."; exit 0; }

    echo "Stelle wieder her ← $infile"

    # Extension sicherstellen
    docker exec "$CONTAINER" \
        psql -U "$DB_USER" -d "$DB_NAME" \
            -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true

    # Dump einspielen (--clean entfernt bestehende Objekte vorher)
    docker exec -i "$CONTAINER" \
        pg_restore -U "$DB_USER" -d "$DB_NAME" \
            --clean \
            --if-exists \
            --no-owner \
            --no-privileges \
            --single-transaction \
        < "$infile"

    count=$(docker exec "$CONTAINER" \
        psql -U "$DB_USER" -d "$DB_NAME" -t -A \
            -c "SELECT COUNT(*) FROM documents;")
    echo "Fertig. $count Dokumente in der Datenbank."
}

# --- Main ---------------------------------------------------------------
case "${1:-}" in
    dump)    do_dump "${2:-}" ;;
    restore) do_restore "${2:-}" ;;
    *)       usage ;;
esac
