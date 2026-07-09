"""
SMS AI Assistant
Receives texts via Twilio, classifies them with Claude, logs to Airtable.

Airtable base needs 3 tables:
  - Tasks    : Date | Task | Status | Original Message
  - Notes    : Date | Note | Tags   | Original Message
  - Expenses : Date | Amount | Category | Description | Original Message
"""

import os
import json
import logging
from datetime import datetime

import requests as http
from flask import Flask, request, Response
from werkzeug.middleware.proxy_fix import ProxyFix
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
AIRTABLE_API_KEY  = os.environ["AIRTABLE_API_KEY"]
AIRTABLE_BASE_ID  = os.environ["AIRTABLE_BASE_ID"]
YOUR_PHONE_NUMBER = os.environ.get("YOUR_PHONE_NUMBER", "")  # optional allowlist

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

AIRTABLE_BASE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json",
}

def airtable_create(table_name: str, fields: dict):
    url = f"{AIRTABLE_BASE_URL}/{http.utils.quote(table_name)}"
    resp = http.post(url, headers=AIRTABLE_HEADERS, json={"fields": fields})
    resp.raise_for_status()
    return resp.json()

def airtable_list(table_name: str, formula: str = None) -> list:
    url = f"{AIRTABLE_BASE_URL}/{http.utils.quote(table_name)}"
    params = {}
    if formula:
        params["filterByFormula"] = formula
    resp = http.get(url, headers=AIRTABLE_HEADERS, params=params)
    resp.raise_for_status()
    return resp.json().get("records", [])

def airtable_update(table_name: str, record_id: str, fields: dict):
    url = f"{AIRTABLE_BASE_URL}/{http.utils.quote(table_name)}/{record_id}"
    resp = http.patch(url, headers=AIRTABLE_HEADERS, json={"fields": fields})
    resp.raise_for_status()
    return resp.json()

# ── Claude parser ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an assistant that parses short text messages and extracts structured data.
The user is an accountant who texts quick notes to themselves.

Classify the message into exactly one category:
  - task     : something to do or follow up on
  - note     : a general note, idea, or piece of information
  - expense  : money spent (look for dollar amounts, "spent", "paid", "cost", "receipt", etc.)
  - hours    : logging billable hours for a client (look for "hours", "add X hours", "log X hours", "X hrs")
  - query    : the user is asking for a summary or list (e.g. "what are my tasks?", "show expenses", "hours for Falls Creek")
  - done     : the user is marking a task complete (e.g. "done: call Mike", "finished the Johnson return")

Respond ONLY with valid JSON matching this schema:

For task:
{"type":"task","task":"<what needs to be done>","reply":"Got it! Added to your tasks."}

For note:
{"type":"note","note":"<the note text>","tags":"<comma-separated keywords or empty string>","reply":"Noted!"}

For expense:
{"type":"expense","amount":"<number only, e.g. 42.50>","category":"<best guess: Meals, Travel, Office, Software, Professional, Other>","description":"<brief description>","reply":"Expense logged: $<amount> for <description>."}

For hours:
{"type":"hours","client":"<client name>","hours":"<number only, e.g. 2.5>","month":"<month and year, e.g. July 2026>","description":"<optional extra detail or empty string>","reply":"Logged <hours> hrs for <client> (<month>)."}

For query:
{"type":"query","query_type":"<tasks|notes|expenses|hours|all>","client_filter":"<client name if asking about a specific client, else empty string>","reply":"<leave empty, will be filled>"}

For done:
{"type":"done","task_hint":"<the task they completed>","reply":"<leave empty, will be filled>"}

Be concise. Infer amounts, hours, and categories intelligently. If a message mentions a client name, include it in the relevant field."""

def parse_message(text: str) -> dict:
    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = response.content[0].text.strip()
    logger.info(f"Claude raw response: {raw!r}")
    # Extract JSON if Claude wrapped it in markdown code fences
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)

# ── Airtable helpers ──────────────────────────────────────────────────────────
def mark_task_done(hint: str) -> str:
    """Find the best-matching open task and mark it Done."""
    records = airtable_list("Tasks", formula="({Status}='Open')")
    if not records:
        return "No open tasks found."

    hint_words = set(hint.lower().split())
    best_record = None
    best_score = 0

    for rec in records:
        task_text = rec["fields"].get("Task", "").lower()
        score = sum(1 for w in hint_words if w in task_text)
        if score > best_score:
            best_score = score
            best_record = rec

    if not best_record or best_score == 0:
        return f"Couldn't find a matching open task for: \"{hint}\""

    airtable_update("Tasks", best_record["id"], {"Status": "Done"})
    task_name = best_record["fields"].get("Task", "that task")
    return f"Marked done: \"{task_name}\""

def build_query_reply(query_type: str, client_filter: str = "") -> str:
    parts = []

    if query_type in ("tasks", "all"):
        records = airtable_list("Tasks", formula="({Status}='Open')")
        if records:
            tasks = [r["fields"].get("Task", "") for r in records[-10:]]
            parts.append("OPEN TASKS:\n" + "\n".join(f"• {t}" for t in tasks))
        else:
            parts.append("No open tasks.")

    if query_type in ("expenses", "all"):
        records = airtable_list("Expenses")
        if records:
            recent = records[-5:]
            total = sum(float(r["fields"].get("Amount", 0)) for r in recent)
            lines = [
                f"• ${float(r['fields'].get('Amount', 0)):.2f} – {r['fields'].get('Description', '')}"
                for r in recent
            ]
            parts.append("RECENT EXPENSES (last 5):\n" + "\n".join(lines) + f"\nTotal shown: ${total:.2f}")
        else:
            parts.append("No expenses logged.")

    if query_type in ("notes", "all"):
        records = airtable_list("Notes")
        if records:
            recent = [r["fields"].get("Note", "") for r in records[-5:]]
            parts.append("RECENT NOTES:\n" + "\n".join(f"• {n}" for n in recent))
        else:
            parts.append("No notes yet.")

    if query_type in ("hours", "all"):
        records = airtable_list("Hours")
        if records:
            # Filter by client if specified
            if client_filter:
                cf = client_filter.lower()
                records = [r for r in records if cf in r["fields"].get("Client", "").lower()]
            if records:
                # Group by client and sum hours
                totals: dict = {}
                for r in records:
                    client = r["fields"].get("Client", "Unknown")
                    hrs = float(r["fields"].get("Hours", 0))
                    totals[client] = totals.get(client, 0) + hrs
                lines = [f"• {c}: {h:.1f} hrs" for c, h in sorted(totals.items())]
                header = f"HOURS{' for ' + client_filter if client_filter else ' BY CLIENT'}:"
                parts.append(header + "\n" + "\n".join(lines))
            else:
                parts.append(f"No hours logged for {client_filter}.")
        else:
            parts.append("No hours logged yet.")

    return "\n\n".join(parts) if parts else "Nothing logged yet."

# ── Main webhook ──────────────────────────────────────────────────────────────
@app.route("/sms", methods=["POST"])
def sms_webhook():
    # Validate the request actually came from Twilio
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    signature = request.headers.get("X-Twilio-Signature", "")
    if not validator.validate(request.url, request.form.to_dict(), signature):
        logger.warning("Invalid Twilio signature — rejected request")
        return Response("Forbidden", status=403)

    sender = request.form.get("From", "")
    body   = request.form.get("Body", "").strip()
    logger.info(f"SMS from {sender}: {body}")

    # Optional: only accept texts from your own number
    if YOUR_PHONE_NUMBER and sender != YOUR_PHONE_NUMBER:
        logger.warning(f"Rejected sender {sender} (expected {YOUR_PHONE_NUMBER})")
        return _reply("Sorry, I only accept messages from the registered number.")

    if not body:
        return _reply("I didn't catch that — try again!")

    logger.info("Calling Claude to parse message...")
    try:
        parsed = parse_message(body)
        logger.info(f"Claude parsed: {parsed}")
    except Exception as e:
        logger.error(f"Parse error: {e}")
        return _reply("Hmm, I had trouble understanding that. Try rephrasing!")

    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_type = parsed.get("type")
        logger.info(f"Writing to Airtable as type: {msg_type}")

        if msg_type == "task":
            airtable_create("Tasks", {
                "Date": now,
                "Task": parsed["task"],
                "Status": "Open",
                "Original Message": body,
            })
            reply = parsed.get("reply", "Task added!")

        elif msg_type == "note":
            airtable_create("Notes", {
                "Date": now,
                "Note": parsed["note"],
                "Tags": parsed.get("tags", ""),
                "Original Message": body,
            })
            reply = parsed.get("reply", "Note saved!")

        elif msg_type == "expense":
            airtable_create("Expenses", {
                "Date": now,
                "Amount": float(parsed.get("amount", 0)),
                "Category": parsed.get("category", "Other"),
                "Description": parsed.get("description", ""),
                "Original Message": body,
            })
            reply = parsed.get("reply", "Expense logged!")

        elif msg_type == "hours":
            airtable_create("Hours", {
                "Date": now,
                "Client": parsed.get("client", ""),
                "Hours": float(parsed.get("hours", 0)),
                "Month": parsed.get("month", ""),
                "Description": parsed.get("description", ""),
                "Original Message": body,
            })
            reply = parsed.get("reply", "Hours logged!")

        elif msg_type == "query":
            reply = build_query_reply(parsed.get("query_type", "all"), parsed.get("client_filter", ""))

        elif msg_type == "done":
            reply = mark_task_done(parsed.get("task_hint", ""))

        else:
            reply = "Logged!"

    except Exception as e:
        logger.error(f"Airtable error: {e}")
        reply = "Saved the message but had trouble writing to Airtable — check the logs."

    return _reply(reply)

def _reply(text: str) -> Response:
    resp = MessagingResponse()
    resp.message(text)
    return Response(str(resp), mimetype="text/xml")

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
