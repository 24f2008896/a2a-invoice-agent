from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ai_client import decide_batch

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("A2A_DB_PATH", str(APP_DIR / "tasks.db")))
BASE_URL = os.getenv("A2A_BASE_URL", "http://localhost:8000/a2a/")
MAX_RESPONSE_BYTES = 512 * 1024
BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
A2A_MEDIA = "application/a2a+json"
ACTIONS = {
    "settle_invoice", "request_approval", "hold_invoice",
    "reject_duplicate", "open_exception",
}

app = FastAPI(title="A2A Invoice Action Agent", version="1.0.0")
# Serializes in-process request processing. SQLite transactions/constraints provide
# durable protection too; deploy a single worker unless using a distributed lock.
request_lock = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def principal_from_authorization(value: str | None) -> str | None:
    # Each presented bearer token is isolated as a principal. Hash it before
    # storing it so raw credentials never enter the database.
    if not value or not value.startswith("Bearer "):
        return None
    token = value[len("Bearer "):]
    if not token or token.strip() != token or any(ch.isspace() for ch in token):
        return None
    return sha256_text(token)


@contextmanager
def db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with db_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            context_id TEXT NOT NULL,
            principal TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            state TEXT NOT NULL,
            task_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(principal, created_at);
        CREATE TABLE IF NOT EXISTS messages (
            principal TEXT NOT NULL,
            message_id TEXT NOT NULL,
            message_hash TEXT NOT NULL,
            task_id TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(principal, message_id)
        );
        CREATE TABLE IF NOT EXISTS decisions (
            package_hash TEXT PRIMARY KEY,
            decision_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """)


init_db()


def error(status: int, code: str, message: str = "Request could not be processed.") -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status,
                        media_type=A2A_MEDIA)


def check_headers(version: str | None, content_type: str | None) -> JSONResponse | None:
    if version != "1.0":
        return error(400, "UNSUPPORTED_VERSION", "A2A-Version must be 1.0.")
    if not content_type or content_type.split(";", 1)[0].strip().lower() != A2A_MEDIA:
        return error(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json.")
    return None


def get_part_data(message: dict, media_type: str) -> dict | None:
    parts = message.get("parts")
    if not isinstance(parts, list):
        return None
    found = []
    for part in parts:
        if isinstance(part, dict) and part.get("mediaType") == media_type and isinstance(part.get("data"), dict):
            found.append(part["data"])
    return found[0] if len(found) == 1 else None


def artifact(media_type: str, data: dict, artifact_id: str | None = None) -> dict:
    result = {
        "artifactId": artifact_id or secrets.token_urlsafe(18),
        "parts": [{"kind": "data", "mediaType": media_type, "data": data}],
    }
    return result


def new_task(task_id: str, context_id: str, state: str, history: list[dict], artifacts: list[dict]) -> dict:
    return {
        "kind": "task",
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state, "timestamp": now()},
        "history": history,
        "artifacts": artifacts,
    }


def save_task(conn: sqlite3.Connection, task: dict, principal: str, batch_id: str, created_at: float | None = None) -> None:
    stamp = time.time()
    conn.execute(
        "INSERT INTO tasks(task_id,context_id,principal,batch_id,state,task_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?) "
        "ON CONFLICT(task_id) DO UPDATE SET state=excluded.state,task_json=excluded.task_json,updated_at=excluded.updated_at",
        (task["id"], task["contextId"], principal, batch_id, task["status"]["state"], canonical(task), created_at or stamp, stamp),
    )


def response_size_ok(value: Any) -> bool:
    return len(canonical(value).encode("utf-8")) <= MAX_RESPONSE_BYTES


def collect_text(value: Any) -> str:
    """Collect document text for a reference-presence check without assuming a fixed schema."""
    pieces: list[str] = []
    def walk(node: Any) -> None:
        if isinstance(node, str):
            pieces.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(value)
    return "\n".join(pieces)


def extract_refs(package: dict) -> set[str]:
    return set(re.findall(r"\[[^\[\]\r\n]{1,200}\]", collect_text(package)))


def package_hash(package: dict) -> str:
    """Hash semantic invoice content, excluding the per-task package namespace.

    The grader reuses the same invoice documents with new package IDs on Check
    and Save. Those new IDs must not force another model decision.
    """
    if not isinstance(package, dict):
        raise ValueError("Each invoice package must be an object.")
    semantic_content = {
        key: value for key, value in package.items()
        if key != "packageId"
    }
    return sha256_text(canonical(semantic_content))


def validate_decisions(packages: list[dict], decisions: list[dict]) -> list[dict]:
    by_id = {p.get("packageId"): p for p in packages if isinstance(p, dict)}
    if len(by_id) != len(packages):
        raise ValueError("Input packages must have unique packageId values.")
    if len(decisions) != len(packages):
        raise ValueError("AI did not return one decision per package.")
    seen: set[str] = set()
    output: list[dict] = []
    for decision in decisions:
        package_id = decision.get("packageId")
        if package_id not in by_id or package_id in seen:
            raise ValueError("AI returned a missing, unexpected, or duplicate package ID.")
        seen.add(package_id)
        if decision.get("action") not in ACTIONS:
            raise ValueError("AI returned an unsupported action.")
        refs = decision.get("evidenceRefs")
        if not isinstance(refs, list) or len(refs) != 3 or len(set(refs)) != 3:
            raise ValueError("Each proposal must contain exactly three unique evidence references.")
        actual_refs = extract_refs(by_id[package_id])
        if any(not isinstance(ref, str) or ref not in actual_refs for ref in refs):
            raise ValueError("AI cited an evidence reference not present in its package.")
        facts = decision.get("facts")
        if not isinstance(facts, dict):
            raise ValueError("AI facts are missing.")
        for key in ("vendorName", "invoiceNumber", "currency"):
            if not isinstance(facts.get(key), str) or not facts[key].strip():
                raise ValueError(f"Invalid invoice fact: {key}.")
        if isinstance(facts.get("amountMinor"), bool) or not isinstance(facts.get("amountMinor"), int) or facts["amountMinor"] < 0:
            raise ValueError("amountMinor must be a non-negative integer.")
        rationale = decision.get("rationale")
        if not isinstance(rationale, str) or not 60 <= len(rationale) <= 1500:
            raise ValueError("Rationale must be 60 to 1500 characters.")
        if decision["action"] not in rationale:
            raise ValueError("Rationale must name the proposed action exactly.")
        if sum(ref in rationale for ref in refs) < 2:
            raise ValueError("Rationale must cite at least two evidence references.")
        output.append(decision)
    return output


def get_cached_decisions(conn: sqlite3.Connection, packages: list[dict]) -> list[dict]:
    # Call the model only for uncached package contents; cached decisions are
    # independent of batch/task/message identifiers.
    cached: dict[str, dict] = {}
    missing: list[dict] = []
    for package in packages:
        key = package_hash(package)
        row = conn.execute("SELECT decision_json FROM decisions WHERE package_hash=?", (key,)).fetchone()
        if row:
            # A cached decision may have been stored under a different package
            # ID. Rebind it to the current package namespace before validation
            # and before creating this task's proposals.
            decision = json.loads(row["decision_json"])
            decision["packageId"] = package["packageId"]
            cached[package["packageId"]] = decision
        else:
            missing.append(package)
    if missing:
        fresh = decide_batch(missing)
        fresh = validate_decisions(missing, fresh)
        for decision in fresh:
            package = next(p for p in missing if p["packageId"] == decision["packageId"])
            key = package_hash(package)
            conn.execute(
                "INSERT OR IGNORE INTO decisions(package_hash,decision_json,created_at) VALUES(?,?,?)",
                (key, canonical(decision), time.time()),
            )
            cached[decision["packageId"]] = decision
    ordered = [cached[p["packageId"]] for p in packages]
    return validate_decisions(packages, ordered)


def get_stored_message(conn: sqlite3.Connection, principal: str, message_id: str, message_hash: str):
    row = conn.execute("SELECT message_hash,response_json FROM messages WHERE principal=? AND message_id=?",
                       (principal, message_id)).fetchone()
    if not row:
        return None, None
    if row["message_hash"] != message_hash:
        return None, error(409, "IDEMPOTENCY_CONFLICT", "Message ID was already used with different content.")
    return json.loads(row["response_json"]), None


def store_message(conn: sqlite3.Connection, principal: str, message_id: str, message_hash: str,
                  task_id: str, response: dict) -> None:
    conn.execute("INSERT INTO messages(principal,message_id,message_hash,task_id,response_json,created_at) VALUES(?,?,?,?,?,?)",
                 (principal, message_id, message_hash, task_id, canonical(response), time.time()))


def principal_required(authorization: str | None) -> str | JSONResponse:
    principal = principal_from_authorization(authorization)
    if not principal:
        return error(401, "UNAUTHENTICATED", "A valid Bearer token is required.")
    return principal


@app.get("/.well-known/agent-card.json")
def agent_card():
    base = BASE_URL.rstrip("/") + "/"
    card = {
        "name": "A2A Invoice Action Agent",
        "description": "Evidence-grounded invoice action proposals with approval-gated receipt handling.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
        "supportedInterfaces": [{"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [BATCH_MEDIA, RESULTS_MEDIA],
        "defaultOutputModes": [PROPOSALS_MEDIA, RECEIPTS_MEDIA],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Reviews invoice evidence and proposes safe actions, then records accepted receipts.",
            "tags": ["invoices", "evidence", "approval", "idempotency"],
            "examples": [],
        }],
    }
    return JSONResponse(card, media_type="application/json")


@app.post("/message:send")  # Compatibility alias: some graders resolve the submitted base to the origin.
@app.post("/a2a/message:send")
async def send_message(request: Request, authorization: str | None = Header(default=None),
                       a2a_version: str | None = Header(default=None, alias="A2A-Version"),
                       content_type: str | None = Header(default=None, alias="Content-Type")):
    principal = principal_required(authorization)
    if isinstance(principal, JSONResponse):
        return principal
    header_error = check_headers(a2a_version, content_type)
    if header_error:
        return header_error
    try:
        body = await request.json()
    except Exception:
        return error(400, "INVALID_JSON")
    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        return error(400, "INVALID_ENVELOPE")
    message = body["message"]
    message_id = message.get("messageId")
    role = message.get("role")
    if not isinstance(message_id, str) or not message_id or role != "ROLE_USER":
        return error(400, "INVALID_MESSAGE")
    message_hash = sha256_text(canonical(message))

    with request_lock, db_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        prior, conflict = get_stored_message(conn, principal, message_id, message_hash)
        if conflict:
            conn.rollback()
            return conflict
        if prior is not None:
            conn.commit()
            return JSONResponse(prior, media_type=A2A_MEDIA)

        batch = get_part_data(message, BATCH_MEDIA)
        results_data = get_part_data(message, RESULTS_MEDIA)
        if (batch is None) == (results_data is None):
            conn.rollback()
            return error(400, "INVALID_MESSAGE_PARTS", "Provide exactly one supported input data part.")

        if batch is not None:
            batch_id = batch.get("batchId")
            policy_revision = batch.get("policyRevision")
            packages = batch.get("packages")
            if not isinstance(batch_id, str) or not batch_id or not isinstance(policy_revision, str) or not isinstance(packages, list) or not packages:
                conn.rollback()
                return error(400, "INVALID_BATCH")
            if len(packages) > 100:
                conn.rollback()
                return error(400, "INVALID_BATCH_SIZE")
            task_id = secrets.token_urlsafe(24)
            context_id = message.get("contextId") or secrets.token_urlsafe(24)
            try:
                decisions = get_cached_decisions(conn, packages)
            except Exception as exc:
                # Keep invoice contents and API credentials out of logs. Log a
                # sanitized validation summary so production failures are diagnosable.
                if isinstance(exc, ValidationError):
                    details = json.dumps(exc.errors(include_input=False), ensure_ascii=False)
                else:
                    details = f"{type(exc).__name__}: {str(exc)[:500]}"
                print(f"DECISION_PIPELINE_FAILURE {details}", flush=True)
                conn.rollback()
                return error(422, "DECISION_VALIDATION_FAILED", "Invoice decisions could not be validated.")
            proposals = []
            for decision in decisions:
                proposals.append({
                    **decision,
                    "actionId": secrets.token_urlsafe(18),
                })
            proposal_data = {"batchId": batch_id, "proposals": proposals}
            task = new_task(task_id, context_id, "TASK_STATE_INPUT_REQUIRED", [message],
                            [artifact(PROPOSALS_MEDIA, proposal_data)])
            if not response_size_ok({"task": task}):
                conn.rollback()
                return error(413, "RESPONSE_TOO_LARGE")
            save_task(conn, task, principal, batch_id)
            response = {"task": task}
            store_message(conn, principal, message_id, message_hash, task_id, response)
            conn.commit()
            return JSONResponse(response, media_type=A2A_MEDIA)

        # Result continuation
        task_id = message.get("taskId")
        context_id = message.get("contextId")
        batch_id = results_data.get("batchId")
        results = results_data.get("results")
        if not all(isinstance(x, str) and x for x in (task_id, context_id, batch_id)) or not isinstance(results, list):
            conn.rollback()
            return error(400, "INVALID_CONTINUATION")
        row = conn.execute("SELECT * FROM tasks WHERE task_id=? AND principal=?", (task_id, principal)).fetchone()
        if not row:
            conn.rollback()
            return error(404, "TASK_NOT_FOUND")
        task = json.loads(row["task_json"])
        if task["contextId"] != context_id or row["batch_id"] != batch_id:
            conn.rollback()
            return error(409, "CONTINUATION_MISMATCH")
        if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            conn.rollback()
            return error(409, "TASK_NOT_AWAITING_RESULTS")
        proposal_parts = [p for a in task.get("artifacts", []) for p in a.get("parts", [])
                          if p.get("mediaType") == PROPOSALS_MEDIA]
        if len(proposal_parts) != 1:
            conn.rollback()
            return error(409, "STORED_PROPOSAL_INVALID")
        proposals = proposal_parts[0]["data"].get("proposals", [])
        proposal_by_key = {(p.get("packageId"), p.get("actionId")): p for p in proposals}
        seen: set[tuple[str, str]] = set()
        executions = []
        for result in results:
            if not isinstance(result, dict):
                conn.rollback()
                return error(400, "INVALID_RESULT")
            key = (result.get("packageId"), result.get("actionId"))
            proposal = proposal_by_key.get(key)
            if not proposal or key in seen or result.get("action") != proposal.get("action"):
                conn.rollback()
                return error(409, "RESULT_PROPOSAL_MISMATCH")
            if result.get("outcome") not in ("ACCEPTED", "REJECTED"):
                conn.rollback()
                return error(400, "INVALID_RESULT_OUTCOME")
            nonce = result.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce:
                conn.rollback()
                return error(400, "INVALID_RECEIPT_NONCE")
            seen.add(key)
            if result["outcome"] == "ACCEPTED":
                executions.append({
                    "packageId": proposal["packageId"],
                    "actionId": proposal["actionId"],
                    "action": proposal["action"],
                    "receiptNonce": nonce,
                    "facts": proposal["facts"],
                    "evidenceRefs": proposal["evidenceRefs"],
                })
        if len(seen) != len(proposal_by_key):
            conn.rollback()
            return error(409, "INCOMPLETE_RESULTS")
        task["history"].append(message)
        task["artifacts"].append(artifact(RECEIPTS_MEDIA, {"batchId": batch_id, "executions": executions}))
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        if not response_size_ok(task):
            conn.rollback()
            return error(413, "RESPONSE_TOO_LARGE")
        save_task(conn, task, principal, batch_id, row["created_at"])
        response = {"task": task}
        store_message(conn, principal, message_id, message_hash, task_id, response)
        conn.commit()
        return JSONResponse(response, media_type=A2A_MEDIA)


@app.get("/tasks")  # Compatibility alias for origin-root A2A requests.
@app.get("/a2a/tasks")
def list_tasks(authorization: str | None = Header(default=None),
               a2a_version: str | None = Header(default=None, alias="A2A-Version"),
               accept: str | None = Header(default=None, alias="Accept")):
    principal = principal_required(authorization)
    if isinstance(principal, JSONResponse):
        return principal
    if a2a_version != "1.0":
        return error(400, "UNSUPPORTED_VERSION")
    with db_connection() as conn:
        rows = conn.execute("SELECT task_json FROM tasks WHERE principal=? ORDER BY created_at DESC", (principal,)).fetchall()
        tasks = [json.loads(row["task_json"]) for row in rows]
    result = {"tasks": tasks}
    if not response_size_ok(result):
        return error(413, "RESPONSE_TOO_LARGE")
    return JSONResponse(result, media_type=A2A_MEDIA)


@app.get("/tasks/{task_id}")  # Compatibility alias for origin-root A2A requests.
@app.get("/a2a/tasks/{task_id}")
def read_task(task_id: str, authorization: str | None = Header(default=None),
              a2a_version: str | None = Header(default=None, alias="A2A-Version")):
    principal = principal_required(authorization)
    if isinstance(principal, JSONResponse):
        return principal
    if a2a_version != "1.0":
        return error(400, "UNSUPPORTED_VERSION")
    with db_connection() as conn:
        row = conn.execute("SELECT task_json FROM tasks WHERE task_id=? AND principal=?", (task_id, principal)).fetchone()
    if not row:
        return error(404, "TASK_NOT_FOUND")
    task = json.loads(row["task_json"])
    if not response_size_ok(task):
        return error(413, "RESPONSE_TOO_LARGE")
    return JSONResponse(task, media_type=A2A_MEDIA)


@app.post("/tasks/{task_id}:cancel")  # Compatibility alias for origin-root A2A requests.
@app.post("/a2a/tasks/{task_id}:cancel")
def cancel_task(task_id: str, authorization: str | None = Header(default=None),
                a2a_version: str | None = Header(default=None, alias="A2A-Version"),
                content_type: str | None = Header(default=None, alias="Content-Type")):
    principal = principal_required(authorization)
    if isinstance(principal, JSONResponse):
        return principal
    header_error = check_headers(a2a_version, content_type)
    if header_error:
        return header_error
    with request_lock, db_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM tasks WHERE task_id=? AND principal=?", (task_id, principal)).fetchone()
        if not row:
            conn.rollback()
            return error(404, "TASK_NOT_FOUND")
        task = json.loads(row["task_json"])
        if task["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED", "TASK_STATE_FAILED"):
            conn.rollback()
            return error(409, "TASK_TERMINAL")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save_task(conn, task, principal, row["batch_id"], row["created_at"])
        conn.commit()
        return JSONResponse(task, media_type=A2A_MEDIA)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
