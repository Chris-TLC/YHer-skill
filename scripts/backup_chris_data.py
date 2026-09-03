"""T0: Back up chris_2026 data (snapshot before the Stage 10 multi-user rework)."""
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

SKILL_DIR = Path(__file__).parent.parent
load_dotenv(SKILL_DIR / ".env")

from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: SUPABASE_URL / SUPABASE_KEY not set")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

tables = {
    "users": "user_id",
    "user_profile": "user_id",
    "query_history": "user_id",
    "memory_summaries": "user_id",
}

backup = {
    "timestamp": datetime.now().isoformat(),
    "description": "Snapshot of chris_2026 data taken before the Stage 10 multi-user rework",
}

total_records = 0
for table_name, key_field in tables.items():
    try:
        result = supabase.table(table_name).select("*").eq(key_field, "chris_2026").execute()
        backup[table_name] = result.data
        n = len(result.data)
        total_records += n
        print(f"  {table_name}: {n} records")
    except Exception as e:
        print(f"  {table_name}: ERROR {e}")
        backup[table_name] = []

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = SKILL_DIR / "data" / "backups" / f"chris_2026_{timestamp}.json"

with open(backup_path, "w", encoding="utf-8") as f:
    json.dump(backup, f, ensure_ascii=False, indent=2, default=str)

print(f"\nBackup complete: {backup_path}")
print(f"   Total {total_records} records (4 tables)")
