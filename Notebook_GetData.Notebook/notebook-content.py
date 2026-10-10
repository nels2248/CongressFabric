# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "39418109-5bb5-4630-a82f-fc5836ac8a08",
# META       "default_lakehouse_name": "Lakehouse1",
# META       "default_lakehouse_workspace_id": "3961b5ec-d383-4029-b4d8-9c4b4acd3d4c",
# META       "known_lakehouses": [
# META         {
# META           "id": "39418109-5bb5-4630-a82f-fc5836ac8a08"
# META         }
# META       ]
# META     },
# META     "warehouse": {
# META       "default_warehouse": "e65dd74a-7993-40d8-9680-d11a6cc54539",
# META       "known_warehouses": [
# META         {
# META           "id": "e65dd74a-7993-40d8-9680-d11a6cc54539",
# META           "type": "Lakewarehouse"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# ============================================================
# Congress 119 — Fabric Medallion Pipeline (Resumable)
# ============================================================
# INSTRUCTIONS:
#   Copy each CELL block below into a separate notebook cell.
#   Run cells one at a time. Each zip is its own cell so a
#   disconnect only loses that one zip — re-run that cell only.
#   Progress is checkpointed to Delta after every zip.
# ============================================================
 
 
# ════════════════════════════════════════════════════════════
# CELL 1 — Config & shared helpers (run once, always first)
# ════════════════════════════════════════════════════════════
 
import io, re, zipfile, requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(notebookutils.fs.mounts())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import json  # needed for etag/seen_files persistence — add if not already imported

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# ── Edit these two lines ──────────────────────────────────
LAKEHOUSE_PATH = "abfss://3961b5ec-d383-4029-b4d8-9c4b4acd3d4c@onelake.dfs.fabric.microsoft.com/39418109-5bb5-4630-a82f-fc5836ac8a08/Files/congress"
MAX_FILES_PER_ZIP = None            # set e.g. 200 to test; None = all
# ─────────────────────────────────────────────────────────

URLS = {
    "hconres": "https://www.govinfo.gov/bulkdata/BILLSTATUS/119/hconres/BILLSTATUS-119-hconres.zip",
    "hjres":   "https://www.govinfo.gov/bulkdata/BILLSTATUS/119/hjres/BILLSTATUS-119-hjres.zip",
    "hr":      "https://www.govinfo.gov/bulkdata/BILLSTATUS/119/hr/BILLSTATUS-119-hr.zip",
    "hres":    "https://www.govinfo.gov/bulkdata/BILLSTATUS/119/hres/BILLSTATUS-119-hres.zip",
    "s":       "https://www.govinfo.gov/bulkdata/BILLSTATUS/119/s/BILLSTATUS-119-s.zip",
    "sconres": "https://www.govinfo.gov/bulkdata/BILLSTATUS/119/sconres/BILLSTATUS-119-sconres.zip",
    "sjres":   "https://www.govinfo.gov/bulkdata/BILLSTATUS/119/sjres/BILLSTATUS-119-sjres.zip",
    "sres":    "https://www.govinfo.gov/bulkdata/BILLSTATUS/119/sres/BILLSTATUS-119-sres.zip",
}

# ── Helpers ───────────────────────────────────────────────

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def get(el, *path, default=None):
    node = el
    for step in path:
        if node is None: return default
        node = node.find(step)
    if node is None or node.text is None: return default
    return node.text.strip() or default

def get_cdata_text(el, tag="text"):
    node = el.find(tag)
    if node is not None and node.text:
        text = re.sub(r"<[^>]+>", " ", node.text)
        return re.sub(r"\s+", " ", text).strip() or None
    cdata = el.find("cdata")
    if cdata is not None:
        node = cdata.find(tag)
        if node is not None and node.text:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", node.text)).strip() or None
    return None

def safe_int(val):
    try: return int(val)
    except: return None



# ── All parsers ───────────────────────────────────────────

import notebookutils  # Fabric-provided; remove if running outside Fabric

def truncate_all(confirm=False):
    if not confirm:
        print("Safety check: call truncate_all(confirm=True) to actually delete.")
        return

    layers_tables = {
        "bronze": ["raw_xml"],
        "silver": ["bills", "sponsors", "cosponsors", "actions", "subjects",
                   "titles", "committees", "summaries", "text_versions",
                   "related_bills", "amendments"],
        "gold":   ["bills_master", "member_activity", "action_timeline",
                   "committee_workload", "bill_ml_features"],
    }

    for layer, tables in layers_tables.items():
        for t in tables:
            path = f"{LAKEHOUSE_PATH}/{layer}/{t}"
            try:
                notebookutils.fs.rm(path, recurse=True)
                print(f"  Deleted: {layer}/{t}")
            except Exception as e:
                print(f"  Skipped {layer}/{t} (not found or already empty): {e}")

    # Wipe checkpoints too — otherwise process_zip thinks everything's already loaded
    checkpoints_path = f"{LAKEHOUSE_PATH}/_checkpoints"
    try:
        notebookutils.fs.rm(checkpoints_path, recurse=True)
        print(f"  Deleted: _checkpoints (all)")
    except Exception as e:
        print(f"  Skipped _checkpoints: {e}")

    print("\n✓ Truncate complete — safe to re-run Cells 2 through 12 from scratch")


truncate_all(confirm=True)

def parse_bill(root, source_file, bill_type_dir):
    bill = root.find("bill")
    if bill is None: return None
    return {
        "source_file": source_file, "bill_type_dir": bill_type_dir,
        "congress": get(bill, "congress"), "bill_type": get(bill, "type"),
        "bill_number": get(bill, "number"),
        "bill_id": f"{get(bill,'congress')}-{get(bill,'type')}-{get(bill,'number')}",
        "title": get(bill, "title"), "introduced_date": get(bill, "introducedDate"),
        "origin_chamber": get(bill, "originChamber"),
        "origin_chamber_code": get(bill, "originChamberCode"),
        "update_date": get(bill, "updateDate"),
        "update_date_incl_text": get(bill, "updateDateIncludingText"),
        "legislation_url": get(bill, "legislationUrl"),
        "latest_action_date": get(bill, "latestAction", "actionDate"),
        "latest_action_time": get(bill, "latestAction", "actionTime"),
        "latest_action_text": get(bill, "latestAction", "text"),
        "policy_area": get(bill, "subjects", "policyArea", "name"),
        "cosponsor_count": safe_int(get(bill, "cosponsors", "count")),
        "cosponsor_count_incl_withdrawn": safe_int(get(bill, "cosponsors", "countIncludingWithdrawnCosponsors")),
        "has_summaries": bill.find("summaries") is not None,
        "has_amendments": bill.find("amendments") is not None,
        "schema_version": get(root, "version"), "ingested_at": now_utc(),
    }

def parse_sponsors(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    sponsors = bill.find("sponsors")
    if sponsors is None: return rows
    for item in sponsors.findall("item"):
        rows.append({"bill_id": bill_id, "bioguide_id": get(item, "bioguideId"),
            "full_name": get(item, "fullName"), "first_name": get(item, "firstName"),
            "middle_name": get(item, "middleName"), "last_name": get(item, "lastName"),
            "party": get(item, "party"), "state": get(item, "state"),
            "district": get(item, "district"), "is_by_request": get(item, "isByRequest")})
    return rows

def parse_cosponsors(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    cs = bill.find("cosponsors")
    if cs is None: return rows
    for item in cs.findall("item"):
        rows.append({"bill_id": bill_id, "bioguide_id": get(item, "bioguideId"),
            "full_name": get(item, "fullName"), "first_name": get(item, "firstName"),
            "middle_name": get(item, "middleName"), "last_name": get(item, "lastName"),
            "party": get(item, "party"), "state": get(item, "state"),
            "district": get(item, "district"), "sponsorship_date": get(item, "sponsorshipDate"),
            "is_original_cosponsor": get(item, "isOriginalCosponsor")})
    return rows

def parse_actions(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    actions = bill.find("actions")
    if actions is None: return rows
    for idx, item in enumerate(actions.findall("item")):
        src = item.find("sourceSystem"); rec = item.find("recordedVotes")
        rows.append({"bill_id": bill_id, "action_seq": idx,
            "action_date": get(item, "actionDate"), "action_time": get(item, "actionTime"),
            "action_code": get(item, "actionCode"), "action_type": get(item, "type"),
            "action_text": get(item, "text"),
            "source_system_name": get(src, "name") if src is not None else None,
            "source_system_code": get(src, "code") if src is not None else None,
            "roll_number": get(rec, "recordedVote", "rollNumber") if rec is not None else None,
            "vote_chamber": get(rec, "recordedVote", "chamber") if rec is not None else None,
            "vote_date": get(rec, "recordedVote", "date") if rec is not None else None,
            "vote_url": get(rec, "recordedVote", "url") if rec is not None else None})
    return rows

def parse_subjects(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    subjects = bill.find("subjects")
    if subjects is None: return rows
    pa = subjects.find("policyArea")
    if pa is not None:
        rows.append({"bill_id": bill_id, "subject_type": "policyArea", "subject_name": get(pa, "name")})
    ls = subjects.find("legislativeSubjects")
    if ls is not None:
        for item in ls.findall("item"):
            name = get(item, "name")
            if name: rows.append({"bill_id": bill_id, "subject_type": "legislativeSubject", "subject_name": name})
    return rows

def parse_titles(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    titles = bill.find("titles")
    if titles is None: return rows
    for item in titles.findall("item"):
        rows.append({"bill_id": bill_id, "title_type": get(item, "titleType"),
            "title_type_code": get(item, "titleTypeCode"), "title": get(item, "title"),
            "chamber": get(item, "chamber"),
            "bill_text_version_name": get(item, "billTextVersionName"),
            "bill_text_version_code": get(item, "billTextVersionCode"),
            "update_date": get(item, "updateDate")})
    return rows

def parse_committees(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    committees = bill.find("committees")
    if committees is None: return rows
    for item in committees.findall("item"):
        sc = get(item, "systemCode"); nm = get(item, "name"); ch = get(item, "chamber")
        activities = item.find("activities")
        if activities:
            for act in activities.findall("item"):
                rows.append({"bill_id": bill_id, "system_code": sc, "committee_name": nm,
                    "chamber": ch, "activity_name": get(act, "name"), "activity_date": get(act, "date")})
        else:
            rows.append({"bill_id": bill_id, "system_code": sc, "committee_name": nm,
                "chamber": ch, "activity_name": None, "activity_date": None})
    return rows

def parse_summaries(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    summaries = bill.find("summaries")
    if summaries is None: return rows
    for s in summaries.findall("summary"):
        rows.append({"bill_id": bill_id, "version_code": get(s, "versionCode"),
            "action_date": get(s, "actionDate"), "action_desc": get(s, "actionDesc"),
            "update_date": get(s, "updateDate"), "summary_text": get_cdata_text(s)})
    return rows

def parse_text_versions(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    tv = bill.find("textVersions")
    if tv is None: return rows
    for item in tv.findall("item"):
        vn = get(item, "type"); vd = get(item, "date")
        formats = item.find("formats")
        if formats:
            for fmt in formats.findall("item"):
                rows.append({"bill_id": bill_id, "version_name": vn, "version_date": vd, "url": get(fmt, "url")})
        else:
            rows.append({"bill_id": bill_id, "version_name": vn, "version_date": vd, "url": None})
    return rows

def parse_related_bills(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    rb = bill.find("relatedBills")
    if rb is None: return rows
    for item in rb.findall("item"):
        rows.append({"bill_id": bill_id,
            "related_congress": get(item, "congress"), "related_type": get(item, "type"),
            "related_number": get(item, "number"),
            "related_bill_id": f"{get(item,'congress')}-{get(item,'type')}-{get(item,'number')}",
            "latest_action_date": get(item, "latestAction", "actionDate"),
            "latest_action_text": get(item, "latestAction", "text"),
            "relationship_type": get(item, "relationshipDetails", "item", "type"),
            "identified_by": get(item, "relationshipDetails", "item", "identifiedBy")})
    return rows

def parse_amendments(root, bill_id):
    rows = []; bill = root.find("bill")
    if bill is None: return rows
    amendments = bill.find("amendments")
    if amendments is None: return rows
    for amend in amendments.findall("amendment"):
        sp = amend.find("sponsors"); sp_item = sp.find("item") if sp is not None else None
        ab = amend.find("amendedBill")
        rows.append({"bill_id": bill_id,
            "amendment_number": get(amend, "number"), "amendment_type": get(amend, "type"),
            "congress": get(amend, "congress"), "chamber": get(amend, "chamber"),
            "submitted_date": get(amend, "submittedDate"), "proposed_date": get(amend, "proposedDate"),
            "purpose": get(amend, "purpose"), "description": get(amend, "description"),
            "sponsor_bioguide_id": get(sp_item, "bioguideId") if sp_item is not None else None,
            "sponsor_full_name": get(sp_item, "fullName") if sp_item is not None else None,
            "sponsor_party": get(sp_item, "party") if sp_item is not None else None,
            "sponsor_state": get(sp_item, "state") if sp_item is not None else None,
            "amended_bill_congress": get(ab, "congress") if ab is not None else None,
            "amended_bill_type": get(ab, "type") if ab is not None else None,
            "amended_bill_number": get(ab, "number") if ab is not None else None,
            "latest_action_date": get(amend, "latestAction", "actionDate"),
            "latest_action_text": get(amend, "latestAction", "text"),
            "update_date": get(amend, "updateDate")})
    return rows

# ── Core per-zip function ─────────────────────────────────

def already_loaded(bill_type_key):
    """
    Legacy one-shot check (kept for the commented-out Cell 9b chunked loader).
    process_zip() no longer calls this — it uses get_checkpoint()/write_checkpoint()
    with ETag/Last-Modified + per-file dedupe instead, so re-runs pick up new bills.
    """
    try:
        checkpoint_path = f"{LAKEHOUSE_PATH}/_checkpoints/{bill_type_key}"
        df = spark.read.format("delta").load(checkpoint_path)
        status = df.collect()[0]["status"]
        if status == "complete":
            print(f"  ✓ {bill_type_key} already loaded — skipping (delete {checkpoint_path} to reload)")
            return True
    except Exception:
        pass
    return False


from pyspark.sql.types import StructType, StructField, StringType

CHECKPOINT_SCHEMA = StructType([
    StructField("bill_type",    StringType(), True),
    StructField("status",       StringType(), True),
    StructField("completed_at", StringType(), True),
    StructField("row_counts",   StringType(), True),
    StructField("etag",         StringType(), True),
    StructField("last_modified", StringType(), True),
    StructField("seen_files",   StringType(), True),
])


def write_checkpoint(bill_type_key, status, metadata=None):
    """
    Persist checkpoint state for bill_type_key as a single-row Delta table
    (overwritten each run) at {LAKEHOUSE_PATH}/_checkpoints/{bill_type_key}.

    metadata may include:
        row_counts:    dict of table -> rows written this run
        etag:          HTTP ETag of the last successfully processed download
        last_modified: HTTP Last-Modified of the last successfully processed download
        seen_files:    list of XML source_file names already ingested (for dedupe)
    """
    metadata = metadata or {}
    checkpoint_path = f"{LAKEHOUSE_PATH}/_checkpoints/{bill_type_key}"
    rows = [{
        "bill_type": bill_type_key,
        "status": status,
        "completed_at": now_utc(),
        "row_counts": json.dumps(metadata.get("row_counts", {})),
        "etag": metadata.get("etag"),
        "last_modified": metadata.get("last_modified"),
        "seen_files": json.dumps(metadata.get("seen_files", [])),
    }]
    spark.createDataFrame(rows, schema=CHECKPOINT_SCHEMA).write.format("delta").mode("overwrite") \
         .option("overwriteSchema", "true").save(checkpoint_path)

def get_checkpoint(bill_type_key):
    """
    Read back the last checkpoint for this key, or None if it's never been written
    (or was written by very old runs before etag/seen_files existed).
    """
    checkpoint_path = f"{LAKEHOUSE_PATH}/_checkpoints/{bill_type_key}"
    try:
        df = spark.read.format("delta").load(checkpoint_path)
    except Exception:
        return None  # table doesn't exist yet — first run

    rows = df.collect()
    if not rows:
        return None

    r = rows[0].asDict()

    try:
        row_counts = json.loads(r.get("row_counts") or "{}")
    except (TypeError, ValueError):
        row_counts = {}  # older checkpoint written with str(dict) instead of JSON

    try:
        seen_files = json.loads(r.get("seen_files") or "[]")
    except (TypeError, ValueError):
        seen_files = []

    return {
        "status": r.get("status"),
        "etag": r.get("etag"),
        "last_modified": r.get("last_modified"),
        "seen_files": seen_files,
        "row_counts": row_counts,
    }


from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, BooleanType, LongType
)

# Explicit schemas — no None-type inference issues
SCHEMAS = {
    "raw_xml": StructType([
        StructField("source_file",     StringType(), True),
        StructField("bill_type_dir",   StringType(), True),
        StructField("file_size_bytes", LongType(),   True),
        StructField("raw_xml",         StringType(), True),
        StructField("ingested_at",     StringType(), True),
    ]),
    "bills": StructType([
        StructField("source_file",                    StringType(),  True),
        StructField("bill_type_dir",                  StringType(),  True),
        StructField("congress",                       StringType(),  True),
        StructField("bill_type",                      StringType(),  True),
        StructField("bill_number",                    StringType(),  True),
        StructField("bill_id",                        StringType(),  True),
        StructField("title",                          StringType(),  True),
        StructField("introduced_date",                StringType(),  True),
        StructField("origin_chamber",                 StringType(),  True),
        StructField("origin_chamber_code",            StringType(),  True),
        StructField("update_date",                    StringType(),  True),
        StructField("update_date_incl_text",          StringType(),  True),
        StructField("legislation_url",                StringType(),  True),
        StructField("latest_action_date",             StringType(),  True),
        StructField("latest_action_time",             StringType(),  True),
        StructField("latest_action_text",             StringType(),  True),
        StructField("policy_area",                    StringType(),  True),
        StructField("cosponsor_count",                IntegerType(), True),
        StructField("cosponsor_count_incl_withdrawn", IntegerType(), True),
        StructField("has_summaries",                  BooleanType(), True),
        StructField("has_amendments",                 BooleanType(), True),
        StructField("schema_version",                 StringType(),  True),
        StructField("ingested_at",                    StringType(),  True),
    ]),
    "sponsors": StructType([
        StructField("bill_id",       StringType(), True),
        StructField("bioguide_id",   StringType(), True),
        StructField("full_name",     StringType(), True),
        StructField("first_name",    StringType(), True),
        StructField("middle_name",   StringType(), True),
        StructField("last_name",     StringType(), True),
        StructField("party",         StringType(), True),
        StructField("state",         StringType(), True),
        StructField("district",      StringType(), True),
        StructField("is_by_request", StringType(), True),
    ]),
    "cosponsors": StructType([
        StructField("bill_id",               StringType(), True),
        StructField("bioguide_id",           StringType(), True),
        StructField("full_name",             StringType(), True),
        StructField("first_name",            StringType(), True),
        StructField("middle_name",           StringType(), True),
        StructField("last_name",             StringType(), True),
        StructField("party",                 StringType(), True),
        StructField("state",                 StringType(), True),
        StructField("district",              StringType(), True),
        StructField("sponsorship_date",      StringType(), True),
        StructField("is_original_cosponsor", StringType(), True),
    ]),
    "actions": StructType([
        StructField("bill_id",            StringType(),  True),
        StructField("action_seq",         IntegerType(), True),
        StructField("action_date",        StringType(),  True),
        StructField("action_time",        StringType(),  True),
        StructField("action_code",        StringType(),  True),
        StructField("action_type",        StringType(),  True),
        StructField("action_text",        StringType(),  True),
        StructField("source_system_name", StringType(),  True),
        StructField("source_system_code", StringType(),  True),
        StructField("roll_number",        StringType(),  True),
        StructField("vote_chamber",       StringType(),  True),
        StructField("vote_date",          StringType(),  True),
        StructField("vote_url",           StringType(),  True),
    ]),
    "subjects": StructType([
        StructField("bill_id",      StringType(), True),
        StructField("subject_type", StringType(), True),
        StructField("subject_name", StringType(), True),
    ]),
    "titles": StructType([
        StructField("bill_id",                 StringType(), True),
        StructField("title_type",              StringType(), True),
        StructField("title_type_code",         StringType(), True),
        StructField("title",                   StringType(), True),
        StructField("chamber",                 StringType(), True),
        StructField("bill_text_version_name",  StringType(), True),
        StructField("bill_text_version_code",  StringType(), True),
        StructField("update_date",             StringType(), True),
    ]),
    "committees": StructType([
        StructField("bill_id",        StringType(), True),
        StructField("system_code",    StringType(), True),
        StructField("committee_name", StringType(), True),
        StructField("chamber",        StringType(), True),
        StructField("activity_name",  StringType(), True),
        StructField("activity_date",  StringType(), True),
    ]),
    "summaries": StructType([
        StructField("bill_id",      StringType(), True),
        StructField("version_code", StringType(), True),
        StructField("action_date",  StringType(), True),
        StructField("action_desc",  StringType(), True),
        StructField("update_date",  StringType(), True),
        StructField("summary_text", StringType(), True),
    ]),
    "text_versions": StructType([
        StructField("bill_id",       StringType(), True),
        StructField("version_name",  StringType(), True),
        StructField("version_date",  StringType(), True),
        StructField("url",           StringType(), True),
    ]),
    "related_bills": StructType([
        StructField("bill_id",            StringType(), True),
        StructField("related_congress",   StringType(), True),
        StructField("related_type",       StringType(), True),
        StructField("related_number",     StringType(), True),
        StructField("related_bill_id",    StringType(), True),
        StructField("latest_action_date", StringType(), True),
        StructField("latest_action_text", StringType(), True),
        StructField("relationship_type",  StringType(), True),
        StructField("identified_by",      StringType(), True),
    ]),
    "amendments": StructType([
        StructField("bill_id",               StringType(), True),
        StructField("amendment_number",      StringType(), True),
        StructField("amendment_type",        StringType(), True),
        StructField("congress",              StringType(), True),
        StructField("chamber",               StringType(), True),
        StructField("submitted_date",        StringType(), True),
        StructField("proposed_date",         StringType(), True),
        StructField("purpose",               StringType(), True),
        StructField("description",           StringType(), True),
        StructField("sponsor_bioguide_id",   StringType(), True),
        StructField("sponsor_full_name",     StringType(), True),
        StructField("sponsor_party",         StringType(), True),
        StructField("sponsor_state",         StringType(), True),
        StructField("amended_bill_congress", StringType(), True),
        StructField("amended_bill_type",     StringType(), True),
        StructField("amended_bill_number",   StringType(), True),
        StructField("latest_action_date",    StringType(), True),
        StructField("latest_action_text",    StringType(), True),
        StructField("update_date",           StringType(), True),
    ]),
}


def write_delta_append(rows, table_name, base_path):
    """Write rows to Delta using an explicit schema — no type inference."""
    if not rows:
        return 0

    schema = SCHEMAS[table_name]

    # Ensure every row has all schema columns (fill missing with None)
    cols = [f.name for f in schema.fields]
    normalized = [{c: row.get(c) for c in cols} for row in rows]

    df = spark.createDataFrame(normalized, schema=schema)
    full_path = f"{base_path}/{table_name}"

    try:
        df.write.format("delta").mode("append").option("mergeSchema", "true").save(full_path)
    except Exception:
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(full_path)

    return len(rows)


def process_zip(bill_type_key, url, force=False):
    """
    Download one zip, parse XML, write/append NEW rows to Silver Delta tables.
    Re-runnable: uses HTTP ETag/Last-Modified to detect if the source changed,
    and skips any XML files already ingested in a prior run so re-running
    only pulls in genuinely new bills.

    force=True bypasses the "did the source change" check and re-downloads
    regardless (still dedupes at the file level).
    """
    checkpoint = get_checkpoint(bill_type_key) or {}
    seen_files = set(checkpoint.get("seen_files", []))

    print(f"\n{'─' * 50}")
    print(f"  Processing: {bill_type_key}")
    print(f"  URL: {url}")

    # --- Check if the source has changed before downloading ---
    if not force and checkpoint.get("status") == "complete":
        try:
            head = requests.head(url, timeout=60, allow_redirects=True)
            head.raise_for_status()
            remote_etag = head.headers.get("ETag")
            remote_lm = head.headers.get("Last-Modified")

            unchanged = (
                (remote_etag and remote_etag == checkpoint.get("etag"))
                or (remote_lm and remote_lm == checkpoint.get("last_modified"))
            )
            if unchanged:
                print(f"  No change since last run — skipping download")
                return
        except requests.RequestException as e:
            # If HEAD fails, fall through and just try the full download
            print(f"  HEAD check failed ({e}), will attempt full download")

    # --- Download ---
    print(f"  Downloading...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=600, stream=False)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  DOWNLOAD FAILED: {e}")
        write_checkpoint(bill_type_key, "download_failed", {})
        raise

    mb = len(resp.content) / 1_048_576
    print(f"{mb:.1f} MB downloaded")

    new_etag = resp.headers.get("ETag")
    new_lm = resp.headers.get("Last-Modified")

    # --- Parse (skip files already ingested) ---
    buckets = {t: [] for t in ["bills", "sponsors", "cosponsors", "actions",
                                "subjects", "titles", "committees", "summaries",
                                "text_versions", "related_bills", "amendments", "raw_xml"]}
    parse_errors = []
    count = 0
    skipped_seen = 0
    newly_seen = []

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        total = len(xml_names)
        print(f"  Parsing {total:,} XML files ({len(seen_files):,} already ingested)...", end=" ", flush=True)

        for xml_name in xml_names:
            source_file = xml_name.split("/")[-1]

            if source_file in seen_files:
                skipped_seen += 1
                continue

            if MAX_FILES_PER_ZIP and count >= MAX_FILES_PER_ZIP:
                break

            try:
                raw = zf.read(xml_name).decode("utf-8", errors="replace")
                root = ET.fromstring(raw)
                bill_row = parse_bill(root, source_file, bill_type_key)
                if not bill_row:
                    continue

                bill_id = bill_row["bill_id"]
                count += 1
                newly_seen.append(source_file)

                buckets["raw_xml"].append({
                    "source_file": source_file,
                    "bill_type_dir": bill_type_key,
                    "file_size_bytes": len(raw),
                    "raw_xml": raw,
                    "ingested_at": now_utc(),
                })
                buckets["bills"].append(bill_row)
                buckets["sponsors"].extend(parse_sponsors(root, bill_id))
                buckets["cosponsors"].extend(parse_cosponsors(root, bill_id))
                buckets["actions"].extend(parse_actions(root, bill_id))
                buckets["subjects"].extend(parse_subjects(root, bill_id))
                buckets["titles"].extend(parse_titles(root, bill_id))
                buckets["committees"].extend(parse_committees(root, bill_id))
                buckets["summaries"].extend(parse_summaries(root, bill_id))
                buckets["text_versions"].extend(parse_text_versions(root, bill_id))
                buckets["related_bills"].extend(parse_related_bills(root, bill_id))
                buckets["amendments"].extend(parse_amendments(root, bill_id))

            except ET.ParseError as e:
                parse_errors.append(f"{xml_name}: {e}")

    print(f"done ({count:,} new bills, {skipped_seen:,} already ingested, {len(parse_errors)} errors)")

    if count == 0:
        print(f"  Nothing new to write — updating checkpoint metadata only")
        write_checkpoint(bill_type_key, "complete", {
            "row_counts": checkpoint.get("row_counts", {}),
            "etag": new_etag,
            "last_modified": new_lm,
            "seen_files": list(seen_files),
        })
        return

    # --- Write Bronze ---
    n_bronze = write_delta_append(buckets["raw_xml"], "raw_xml", f"{LAKEHOUSE_PATH}/bronze")

    # --- Write Silver ---
    row_counts = {"raw_xml": n_bronze}
    for table in ["bills", "sponsors", "cosponsors", "actions", "subjects",
                  "titles", "committees", "summaries", "text_versions",
                  "related_bills", "amendments"]:
        n = write_delta_append(buckets[table], table, f"{LAKEHOUSE_PATH}/silver")
        row_counts[table] = n
        print(f"    silver.{table:<28} {n:>8,} rows appended")

    write_checkpoint(bill_type_key, "complete", {
        "row_counts": row_counts,
        "etag": new_etag,
        "last_modified": new_lm,
        "seen_files": list(seen_files.union(newly_seen)),
    })
    print(f"  ✓ {bill_type_key} complete — {count:,} new bills, checkpoint saved")


print("✓ Cell 1 loaded — helpers & config ready")
print(f"  LAKEHOUSE_PATH = {LAKEHOUSE_PATH}")
print(f"  MAX_FILES_PER_ZIP = {MAX_FILES_PER_ZIP}")


# ════════════════════════════════════════════════════════════
# CELL 2 — Load hconres  (~30 bills, fast)
# ════════════════════════════════════════════════════════════

process_zip("hconres", URLS["hconres"])


# ════════════════════════════════════════════════════════════
# CELL 3 — Load hjres
# ════════════════════════════════════════════════════════════

process_zip("hjres", URLS["hjres"])


# ════════════════════════════════════════════════════════════
# CELL 4 — Load hres
# ════════════════════════════════════════════════════════════

process_zip("hres", URLS["hres"])


# ════════════════════════════════════════════════════════════
# CELL 5 — Load sconres
# ════════════════════════════════════════════════════════════

process_zip("sconres", URLS["sconres"])


# ════════════════════════════════════════════════════════════
# CELL 6 — Load sjres
# ════════════════════════════════════════════════════════════

process_zip("sjres", URLS["sjres"])


# ════════════════════════════════════════════════════════════
# CELL 7 — Load sres
# ════════════════════════════════════════════════════════════

process_zip("sres", URLS["sres"])


# ════════════════════════════════════════════════════════════
# CELL 8 — Load s  (Senate bills, moderate size)
# ════════════════════════════════════════════════════════════

process_zip("s", URLS["s"])


# ════════════════════════════════════════════════════════════
# CELL 9 — Load hr  (House bills — LARGEST, ~10k+ files)
#   If this still disconnects, set MAX_FILES_PER_ZIP = 2000
#   in Cell 1 and re-run cells 1 + 9 multiple times,
#   temporarily changing the URL to avoid re-downloading.
#   Or use the chunked loader in Cell 9b below instead.
# ════════════════════════════════════════════════════════════

process_zip("hr", URLS["hr"])


# ════════════════════════════════════════════════════════════
# CELL 9b — CHUNKED loader for hr (use instead of Cell 9
#            if Cell 9 keeps disconnecting)
# ════════════════════════════════════════════════════════════
#
# This downloads hr once, saves to Lakehouse Files, then
# processes it in chunks of CHUNK_SIZE bills at a time.
# Each chunk writes to Delta and checkpoints independently.
#
# UNCOMMENT and run this cell instead of Cell 9 if needed.

# CHUNK_SIZE = 1000   # bills per batch
#
# def process_hr_chunked(chunk_size=1000):
#     bill_type_key = "hr"
#     url = URLS["hr"]
#
#     # Check if fully done
#     if already_loaded(bill_type_key):
#         return
#
#     print(f"Downloading hr zip...")
#     resp = requests.get(url, timeout=600)
#     resp.raise_for_status()
#     print(f"  {len(resp.content)/1_048_576:.1f} MB downloaded")
#
#     with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
#         xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
#
#     # Find which chunk to start from
#     start_chunk = 0
#     try:
#         cp = spark.read.format("delta").load(f"{LAKEHOUSE_PATH}/_checkpoints/hr_chunk").collect()
#         start_chunk = max(r["chunk"] for r in cp) + 1
#         print(f"  Resuming from chunk {start_chunk} ({start_chunk * chunk_size:,} bills already loaded)")
#     except Exception:
#         pass
#
#     total_chunks = (len(xml_names) + chunk_size - 1) // chunk_size
#     print(f"  Total: {len(xml_names):,} files → {total_chunks} chunks of {chunk_size}")
#
#     with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
#         for chunk_idx in range(start_chunk, total_chunks):
#             chunk_files = xml_names[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
#             print(f"\n  Chunk {chunk_idx+1}/{total_chunks} ({len(chunk_files)} files)...", end=" ")
#
#             buckets = {t: [] for t in ["bills","sponsors","cosponsors","actions",
#                                        "subjects","titles","committees","summaries",
#                                        "text_versions","related_bills","amendments","raw_xml"]}
#             for xml_name in chunk_files:
#                 try:
#                     raw = zf.read(xml_name).decode("utf-8", errors="replace")
#                     root = ET.fromstring(raw)
#                     bill_row = parse_bill(root, xml_name.split("/")[-1], bill_type_key)
#                     if not bill_row: continue
#                     bill_id = bill_row["bill_id"]
#                     buckets["raw_xml"].append({"source_file": xml_name.split("/")[-1],
#                         "bill_type_dir": bill_type_key, "file_size_bytes": len(raw),
#                         "raw_xml": raw, "ingested_at": now_utc()})
#                     buckets["bills"].append(bill_row)
#                     buckets["sponsors"].extend(parse_sponsors(root, bill_id))
#                     buckets["cosponsors"].extend(parse_cosponsors(root, bill_id))
#                     buckets["actions"].extend(parse_actions(root, bill_id))
#                     buckets["subjects"].extend(parse_subjects(root, bill_id))
#                     buckets["titles"].extend(parse_titles(root, bill_id))
#                     buckets["committees"].extend(parse_committees(root, bill_id))
#                     buckets["summaries"].extend(parse_summaries(root, bill_id))
#                     buckets["text_versions"].extend(parse_text_versions(root, bill_id))
#                     buckets["related_bills"].extend(parse_related_bills(root, bill_id))
#                     buckets["amendments"].extend(parse_amendments(root, bill_id))
#                 except ET.ParseError:
#                     pass
#
#             write_delta_append(buckets["raw_xml"], "raw_xml", f"{LAKEHOUSE_PATH}/bronze")
#             for t in ["bills","sponsors","cosponsors","actions","subjects","titles",
#                       "committees","summaries","text_versions","related_bills","amendments"]:
#                 write_delta_append(buckets[t], t, f"{LAKEHOUSE_PATH}/silver")
#
#             # Save chunk checkpoint
#             spark.createDataFrame([{"chunk": chunk_idx, "files": len(chunk_files),
#                                     "completed_at": now_utc()}]) \
#                  .write.format("delta").mode("append") \
#                  .save(f"{LAKEHOUSE_PATH}/_checkpoints/hr_chunk")
#             print(f"✓ chunk {chunk_idx+1} saved")
#
#     write_checkpoint("hr", "complete", {"chunked": True})
#     print(f"\n✓ hr fully loaded via chunked mode")
#
# process_hr_chunked(CHUNK_SIZE)


# ════════════════════════════════════════════════════════════
# CELL 10 — Build GOLD layer  (run after all Silver is loaded)
# ════════════════════════════════════════════════════════════

print("Building Gold layer...")

# Register Silver tables as temp views
silver_tables = ["bills","sponsors","cosponsors","actions","subjects","titles",
                 "committees","summaries","text_versions","related_bills","amendments"]
for t in silver_tables:
    spark.read.format("delta").load(f"{LAKEHOUSE_PATH}/silver/{t}").createOrReplaceTempView(f"silver_{t}")

gold_queries = {
    "bills_master": """
        SELECT b.bill_id, b.congress, b.bill_type, b.bill_number, b.title,
               b.introduced_date, b.origin_chamber, b.policy_area,
               b.latest_action_date, b.latest_action_text, b.legislation_url,
               b.cosponsor_count, b.has_summaries, b.has_amendments,
               s.bioguide_id AS sponsor_bioguide_id, s.full_name AS sponsor_full_name,
               s.party AS sponsor_party, s.state AS sponsor_state, s.district AS sponsor_district,
               DATEDIFF(current_date(), TO_DATE(b.introduced_date, 'yyyy-MM-dd')) AS days_since_intro,
               act.action_count
        FROM silver_bills b
        LEFT JOIN silver_sponsors s ON b.bill_id = s.bill_id
        LEFT JOIN (SELECT bill_id, COUNT(*) AS action_count FROM silver_actions GROUP BY bill_id) act
            ON b.bill_id = act.bill_id
    """,
    "member_activity": """
        SELECT bioguide_id, full_name, party, state, district,
               COUNT(*) AS bills_sponsored, CAST(NULL AS BIGINT) AS bills_cosponsored, 'sponsor' AS role
        FROM silver_sponsors WHERE bioguide_id IS NOT NULL
        GROUP BY bioguide_id, full_name, party, state, district
        UNION ALL
        SELECT bioguide_id, full_name, party, state, district,
               CAST(NULL AS BIGINT), COUNT(*), 'cosponsor'
        FROM silver_cosponsors WHERE bioguide_id IS NOT NULL
        GROUP BY bioguide_id, full_name, party, state, district
    """,
    "action_timeline": """
        SELECT a.bill_id, b.bill_type, b.bill_number, b.title, b.origin_chamber, b.policy_area,
               a.action_seq, a.action_date, a.action_time, a.action_type, a.action_text,
               a.source_system_name, a.action_code, a.roll_number, a.vote_chamber
        FROM silver_actions a JOIN silver_bills b ON a.bill_id = b.bill_id
    """,
    "committee_workload": """
        SELECT system_code, committee_name, chamber,
               COUNT(DISTINCT bill_id) AS bill_count, COUNT(*) AS total_activities,
               MIN(activity_date) AS earliest_activity, MAX(activity_date) AS latest_activity
        FROM silver_committees
        GROUP BY system_code, committee_name, chamber
        ORDER BY bill_count DESC
    """,
    "bill_ml_features": """
        SELECT b.bill_id, b.congress, b.bill_type, b.origin_chamber, b.policy_area,
               b.introduced_date, 
               coalesce(cosp.cosponsor_count,0) as cosponsor_count, 
               coalesce(act.action_count,0) as action_count, 
               coalesce(com.committee_count,0) as committee_count,
               coalesce(amn.amendment_count,0) as amendment_count,
               coalesce(rb.related_bills_count,0) as related_bills_count,
               coalesce(sub.subjects_count,0) as subjects_count,
               coalesce(tv.text_versions_count,0) as text_versions_count,
               coalesce(tit.title_count,0) as title_count,
               CASE WHEN sp.party = 'R' THEN 1 ELSE 0 END     AS sponsor_is_republican,
               CASE WHEN sp.party = 'D' THEN 1 ELSE 0 END     AS sponsor_is_democrat,
               CASE WHEN sp.district IS NULL THEN 1 ELSE 0 END AS sponsor_is_senator,
               CASE
                   WHEN lower(b.latest_action_text) LIKE '%enacted%'                THEN 'enacted'
                   WHEN lower(b.latest_action_text) LIKE '%signed by president%'    THEN 'enacted'
                   WHEN lower(b.latest_action_text) LIKE '%became public law%'      THEN 'enacted'
                   WHEN lower(b.latest_action_text) LIKE '%passed senate%'          THEN 'passed_senate'
                   WHEN lower(b.latest_action_text) LIKE '%passed house%'           THEN 'passed_house'
                   WHEN lower(b.latest_action_text) LIKE '%received in the senate%' THEN 'passed_house'
                   WHEN lower(b.latest_action_text) LIKE '%referred to%'            THEN 'referred'
                   ELSE 'other'
               END AS bill_status_label 
        FROM silver_bills b
        LEFT JOIN silver_sponsors sp ON b.bill_id = sp.bill_id
        LEFT JOIN (SELECT bill_id, COUNT(*) AS action_count FROM silver_actions GROUP BY bill_id) act ON b.bill_id = act.bill_id
        LEFT JOIN (SELECT bill_id, COUNT(DISTINCT system_code) AS committee_count FROM silver_committees GROUP BY bill_id) com ON b.bill_id = com.bill_id
        LEFT JOIN (SELECT bill_id, COUNT(*) AS cosponsor_count FROM silver_cosponsors GROUP BY bill_id) cosp ON b.bill_id = cosp.bill_id
        LEFT JOIN (SELECT bill_id, COUNT(*) AS amendment_count FROM silver_amendments GROUP BY bill_id) amn ON b.bill_id = amn.bill_id
        LEFT JOIN (SELECT bill_id, COUNT(*) AS related_bills_count FROM silver_related_bills GROUP BY bill_id) rb ON b.bill_id = rb.bill_id
        LEFT JOIN (SELECT bill_id, COUNT(*) AS subjects_count FROM silver_subjects GROUP BY bill_id) sub ON b.bill_id = sub.bill_id
        LEFT JOIN (SELECT bill_id, COUNT(*) AS text_versions_count FROM silver_text_versions GROUP BY bill_id) tv ON b.bill_id = tv.bill_id
        LEFT JOIN (SELECT bill_id, COUNT(*) AS title_count FROM silver_titles GROUP BY bill_id) tit ON b.bill_id = tit.bill_id
    """,
}

for name, sql in gold_queries.items():
    df = spark.sql(sql)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
       .save(f"{LAKEHOUSE_PATH}/gold/{name}")
    n = df.count()
    print(f"  gold.{name:<30} {n:>10,} rows")

print("✓ Gold layer complete")


# ════════════════════════════════════════════════════════════
# CELL 11 — Register all tables in Lakehouse metastore
#            (makes them visible in SQL Endpoint & Power BI)
# ════════════════════════════════════════════════════════════

for db in ["congress_bronze", "congress_silver", "congress_gold"]:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")

registrations = {
    "congress_bronze": ["raw_xml"],
    "congress_silver": ["bills","sponsors","cosponsors","actions","subjects","titles",
                        "committees","summaries","text_versions","related_bills","amendments"],
    "congress_gold":   ["bills_master","member_activity","action_timeline",
                        "committee_workload","bill_ml_features"],
}

for db, tables in registrations.items():
    layer = db.split("_")[1]
    for t in tables:
        path = f"{LAKEHOUSE_PATH}/{layer}/{t}"
        spark.sql(f"CREATE TABLE IF NOT EXISTS {db}.{t} USING DELTA LOCATION '{path}'")
        print(f"  Registered: {db}.{t}")

print("\n✓ All tables registered in metastore")


# ════════════════════════════════════════════════════════════
# CELL 12 — Validation summary
# ════════════════════════════════════════════════════════════

print("=" * 55)
print("VALIDATION SUMMARY")
print("=" * 55)

spark.sql("""
    SELECT bill_type, COUNT(*) AS bills
    FROM congress_silver.bills
    GROUP BY bill_type ORDER BY bills DESC
""").show()

spark.sql("SELECT COUNT(*) AS total_actions FROM congress_silver.actions").show()
spark.sql("SELECT COUNT(*) AS total_cosponsors FROM congress_silver.cosponsors").show()
spark.sql("SELECT bill_status_label, COUNT(*) AS cnt FROM congress_gold.bill_ml_features GROUP BY 1 ORDER BY 2 DESC").show()

print("Pipeline complete. Tables available in:")
print("  • SQL Analytics Endpoint: congress_silver.bills, congress_gold.bills_master, etc.")
print("  • Power BI DirectLake: point at congress_gold schema")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

LAKEHOUSE_PATH = "abfss://3961b5ec-d383-4029-b4d8-9c4b4acd3d4c@onelake.dfs.fabric.microsoft.com/39418109-5bb5-4630-a82f-fc5836ac8a08/Files/congress"


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

for db in ["congress_bronze", "congress_silver", "congress_gold"]:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")

registrations = {
    "congress_bronze": ["raw_xml"],
    "congress_silver": ["bills","sponsors","cosponsors","actions","subjects","titles",
                        "committees","summaries","text_versions","related_bills","amendments"],
    "congress_gold":   ["bills_master","member_activity","action_timeline",
                        "committee_workload","bill_ml_features"],
}

for db, tables in registrations.items():
    layer = db.split("_")[1]
    for t in tables:
        path = f"{LAKEHOUSE_PATH}/{layer}/{t}"
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {db}.{t}
            USING DELTA
            LOCATION '{path}'
        """)
        print(f"  Registered: {db}.{t}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
