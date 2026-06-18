"""
Excel Q&A Telegram Bot — Text-to-Pandas Architecture
-----------------------------------------------------
Professional code-generation approach: sends ONLY the schema (~500 chars)
to Gemini, which generates pandas code that runs locally. Cuts token
usage by ~99% compared to dumping all data into the prompt.

SETUP:
1. pip install -r requirements.txt
2. Set TELEGRAM_BOT_TOKEN and GEMINI_API_KEY in .env
3. Put your Excel file in this folder and update EXCEL_FILE_PATH
4. Run: python bot.py
"""

import os
import difflib
import re
import time
import logging
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ── CONFIG ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
EXCEL_FILE_PATH = os.environ.get("EXCEL_FILE_PATH", "naxcuure(3).xlsx")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")

if not TELEGRAM_BOT_TOKEN or not OPENROUTER_API_KEY:
    raise ValueError(
        "Missing TELEGRAM_BOT_TOKEN or OPENROUTER_API_KEY in environment variables."
    )

# ── LOGGING ───────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── DATA LOADING ──────────────────────────────────────────────────────────
def load_data(path: str) -> dict[str, pd.DataFrame]:
    """Load every sheet in the Excel workbook into a dict of DataFrames."""
    xls = pd.ExcelFile(path, engine="openpyxl")
    dataframes = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        # Drop fully-empty columns and rows
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        # Auto-clean: convert comma-formatted number strings to actual numbers.
        # Excel files often store quantities like "1,60,000.00" as text.
        for col in df.columns:
            if df[col].dtype == "object":
                # Strip trailing/leading whitespace from all string columns
                df[col] = df[col].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )
                # Try converting comma-separated number strings to float
                sample = df[col].dropna().head(20)
                converted = False
                if len(sample) > 0:
                    numeric_count = 0
                    for val in sample:
                        if isinstance(val, str):
                            cleaned = val.replace(",", "").strip()
                            try:
                                float(cleaned)
                                numeric_count += 1
                            except ValueError:
                                pass
                    # If >70% of sampled values look numeric, convert the column
                    if numeric_count / len(sample) > 0.7:
                        df[col] = pd.to_numeric(
                            df[col].apply(
                                lambda x: str(x).replace(",", "").strip()
                                if pd.notna(x)
                                else x
                            ),
                            errors="coerce",
                        )
                        converted = True

                # Auto-detect date columns (only if not already converted to numeric)
                if not converted and df[col].dtype == "object":
                    date_sample = df[col].dropna().head(10)
                    if len(date_sample) > 0:
                        try:
                            parsed = pd.to_datetime(date_sample, errors="coerce")
                            if parsed.notna().sum() / len(date_sample) > 0.7:
                                df[col] = pd.to_datetime(df[col], errors="coerce")
                                logger.info(f"  Converted '{col}' to datetime")
                                converted = True
                        except Exception:
                            pass

                # Normalize categorical string columns to UPPERCASE
                # (fixes inconsistencies like 'COMPLETED' vs 'completed')
                if not converted and df[col].dtype == "object":
                    nunique = df[col].nunique()
                    if nunique <= 20:  # categorical-like columns only
                        df[col] = df[col].apply(
                            lambda x: x.upper() if isinstance(x, str) else x
                        )
        dataframes[sheet] = df
        logger.info(
            f"Loaded sheet '{sheet}': {len(df)} rows x {len(df.columns)} cols"
        )
    return dataframes


# ── SCHEMA EXTRACTION ────────────────────────────────────────────────────
def get_schema_prompt(dataframes: dict[str, pd.DataFrame]) -> str:
    """Build a compact schema description for the LLM — column names, types,
    unique values for categoricals, and 2 sample rows. Typically ~500 chars
    instead of 122,000."""
    parts = []
    for sheet_name, df in dataframes.items():
        parts.append(f"=== SHEET: {sheet_name} ({len(df)} rows) ===")
        parts.append("COLUMNS:")

        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            info = f"  - {col} ({dtype})"

            # For columns with few unique values, list them all (helps the LLM
            # generate correct filters without seeing the raw data).
            if nunique <= 25 and dtype == "object":
                unique_vals = df[col].dropna().unique().tolist()
                info += f" — values: {unique_vals}"
            elif dtype in ("float64", "int64"):
                info += f" — range: [{df[col].min()}, {df[col].max()}]"
            elif "datetime" in dtype:
                info += f" — range: [{df[col].min()}, {df[col].max()}]"

            parts.append(info)

        # 2 sample rows so the LLM sees the data shape
        parts.append("\nSAMPLE ROWS (first 2):")
        sample = df.head(2).to_string(index=False)
        parts.append(sample)
        parts.append("")

    return "\n".join(parts)


# ── SAFE CODE EXECUTION ──────────────────────────────────────────────────
# Whitelist of safe names — no file I/O, no imports, no network access.
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

# Patterns that should NEVER appear in generated code
BLOCKED_PATTERNS = [
    r"\bimport\b",
    r"\b__\w+__\b",  # dunder attributes
    r"\bexec\b",
    r"\beval\b",
    r"\bopen\b",
    r"\bos\b",
    r"\bsys\b",
    r"\bsubprocess\b",
    r"\bglobals\b",
    r"\blocals\b",
    r"\bcompile\b",
    r"\bgetattr\b",
    r"\bsetattr\b",
    r"\bdelattr\b",
]


def execute_safely(
    code: str, dataframes: dict[str, pd.DataFrame]
) -> tuple[bool, str]:
    """Execute LLM-generated pandas code in a restricted sandbox.

    Returns (success: bool, result_or_error: str).

    The code must store its final answer in a variable called `result`.
    Available variables: df (main sheet), all sheet DataFrames by name,
    pd (pandas), np (numpy).
    """
    # Security checks
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked unsafe pattern in generated code: {pattern}"

    # Build the sandbox namespace
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "datetime": datetime,
        "timedelta": timedelta,
    }

    # Make all sheets available — main sheet as `df`, others by clean name
    sheet_names = list(dataframes.keys())
    if sheet_names:
        namespace["df"] = dataframes[sheet_names[0]].copy()
        for name, frame in dataframes.items():
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
            namespace[clean_name] = frame.copy()

    try:
        exec(code, namespace)  # noqa: S102
        result = namespace.get("result", None)
        if result is None:
            return False, "Code did not set a 'result' variable."

        # Convert DataFrames/Series to string for the formatting step
        if isinstance(result, (pd.DataFrame, pd.Series)):
            if isinstance(result, pd.DataFrame) and len(result) > 50:
                result_str = (
                    result.head(50).to_string(index=False)
                    + f"\n... ({len(result)} rows total, showing first 50)"
                )
            else:
                result_str = result.to_string(index=False)
        else:
            result_str = str(result)

        return True, result_str

    except Exception as e:
        return False, f"Execution error: {type(e).__name__}: {e}"


# ── RESPONSE CACHE ────────────────────────────────────────────────────────
class ResponseCache:
    """Simple TTL-based cache to avoid redundant API calls."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[str, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def _key(self, question: str) -> str:
        normalized = question.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, question: str) -> str | None:
        key = self._key(question)
        if key in self._cache:
            answer, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._ttl:
                logger.info("Cache hit for question")
                return answer
            del self._cache[key]
        return None

    def put(self, question: str, answer: str):
        self._cache[self._key(question)] = (answer, datetime.now())

    def clear(self):
        self._cache.clear()


# ── PER-USER RATE LIMITER ─────────────────────────────────────────────────
class UserRateLimiter:
    """Allow max N questions per user per window (seconds)."""

    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        self._max = max_requests
        self._window = timedelta(seconds=window_seconds)
        self._history: dict[int, list[datetime]] = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        now = datetime.now()
        cutoff = now - self._window
        # Prune old entries
        self._history[user_id] = [
            t for t in self._history[user_id] if t > cutoff
        ]
        if len(self._history[user_id]) >= self._max:
            return False
        self._history[user_id].append(now)
        return True

    def seconds_until_next(self, user_id: int) -> int:
        if not self._history[user_id]:
            return 0
        oldest = min(self._history[user_id])
        wait = (oldest + self._window) - datetime.now()
        return max(0, int(wait.total_seconds()) + 1)


# ── OPENROUTER SETUP ─────────────────────────────────────────────────────────
llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Load data once at startup
DATAFRAMES = load_data(EXCEL_FILE_PATH)
SCHEMA_PROMPT = get_schema_prompt(DATAFRAMES)
logger.info(f"Schema prompt: {len(SCHEMA_PROMPT)} characters")

# Instances
cache = ResponseCache(ttl_seconds=300)  # 5-minute cache
rate_limiter = UserRateLimiter(max_requests=5, window_seconds=60)
user_histories: dict[int, list[tuple[str, str]]] = {}


# ── FUZZY AUTO-CORRECTION ─────────────────────────────────────────────────
# Collect all unique string values from the data so we can auto-correct typos
# BEFORE the question reaches Gemini. This runs locally, zero API cost.
def _build_known_values(dataframes: dict[str, pd.DataFrame]) -> set[str]:
    """Extract all unique string values from categorical columns."""
    values = set()
    for df in dataframes.values():
        for col in df.columns:
            if df[col].dtype == "object":
                nunique = df[col].nunique()
                # Only collect values from columns with a reasonable number of
                # unique entries (categorical-like). Skip free-text columns
                # like REMARKS or BATCH NO.
                if nunique <= 30:
                    for val in df[col].dropna().unique():
                        val_str = str(val).strip()
                        if len(val_str) >= 3:  # skip very short values
                            values.add(val_str)
    return values


KNOWN_VALUES = _build_known_values(DATAFRAMES)
# Build a lowercase lookup map: lowercase → original
KNOWN_VALUES_LOWER = {v.lower(): v for v in KNOWN_VALUES}
logger.info(f"Fuzzy matcher loaded with {len(KNOWN_VALUES)} known values")


def fuzzy_correct_question(question: str) -> tuple[str, list[str]]:
    """Auto-correct misspelled data values in the user's question.

    Uses difflib to find close matches against known product names, countries,
    statuses, stages, etc. Returns the corrected question and a list of
    corrections made (for logging/transparency).

    Examples:
        'bolnl tablet status' → 'BOLNOL TABLET status'
        'how many running in uzbek' → 'how many RUNNING in UZBEKISTAN'
        'cefixl batches' → 'CEFIXOL-200 batches'
    """
    corrections = []
    words = question.split()
    corrected_words = list(words)  # copy

    # Try matching progressively longer word groups (1-word, 2-word, 3-word)
    # to handle multi-word values like "BOLNOL TABLET" or "ALU ALU M/C"
    i = 0
    while i < len(words):
        matched = False
        # Try 3-word, then 2-word, then 1-word phrases
        for n_words in [3, 2, 1]:
            if i + n_words > len(words):
                continue
            phrase = " ".join(words[i : i + n_words]).strip()
            phrase_lower = phrase.lower()

            # Exact match (case-insensitive) — just fix the case
            if phrase_lower in KNOWN_VALUES_LOWER:
                original = KNOWN_VALUES_LOWER[phrase_lower]
                if phrase != original:
                    for j in range(n_words):
                        corrected_words[i + j] = "" if j > 0 else original
                    corrections.append(f"'{phrase}' -> '{original}'")
                matched = True
                i += n_words
                break

            # Fuzzy match — only for single words >= 4 chars to avoid
            # false positives on short common words like "in", "the", etc.
            if n_words == 1 and len(phrase) >= 4:
                close = difflib.get_close_matches(
                    phrase_lower,
                    KNOWN_VALUES_LOWER.keys(),
                    n=1,
                    cutoff=0.6,  # 60% similarity threshold
                )
                if close:
                    original = KNOWN_VALUES_LOWER[close[0]]
                    corrected_words[i] = original
                    corrections.append(f"'{phrase}' -> '{original}' (fuzzy)")
                    matched = True
                    # If the matched value is multi-word (e.g. "BOLNOL TABLET"),
                    # skip subsequent words that are already part of it so we
                    # don't get duplicates like "VRAGGRIPP TABLET tablet".
                    match_words = original.lower().split()
                    skip = 0
                    for mw in match_words[1:]:
                        if (i + 1 + skip) < len(words) and words[i + 1 + skip].lower() == mw:
                            corrected_words[i + 1 + skip] = ""
                            skip += 1
                    i += 1 + skip
                    break

        if not matched:
            i += 1

    corrected = " ".join(w for w in corrected_words if w).strip()
    return corrected, corrections


# ── CODE GENERATION PROMPT ────────────────────────────────────────────────
CODE_GEN_SYSTEM = """\
You are a strict data analyst assistant. Given a spreadsheet schema and a \
user question, you MUST do one of two things:

1. If the question CAN be answered using the spreadsheet data below, write \
   Python pandas code to answer it.
2. If the question CANNOT be answered from the spreadsheet data (e.g. general \
   knowledge, opinions, predictions, anything not in the columns), respond \
   with EXACTLY the single word: NOT_RELEVANT

EXAMPLES OF NOT_RELEVANT QUESTIONS:
- "What is the weather today?"
- "Who is the prime minister?"
- "What is the market price of these drugs?"
- "Will production increase next month?"
- "Tell me a joke"
- Any greeting like "hi", "hello", "how are you"

CRITICAL — FUZZY & CASE-INSENSITIVE MATCHING:
The user may type in lowercase, have typos, or use partial names. You MUST:
- ALWAYS use case-insensitive matching. NEVER use exact `==` for string columns.
- Use `df[col].str.contains("keyword", case=False, na=False)` for filtering.
- For partial/misspelled names, match the CLOSEST value from the schema's \
  listed unique values. For example:
  - "bolnol" → matches "BOLNOL TABLET"
  - "running" → matches "RUNNING"
  - "uzbek" → matches "UZBEKISTAN"
  - "cefixol" → matches "CEFIXOL-200"
- When the user mentions a value, find the best matching value from the schema \
  and use `.str.contains()` with the KEY PART of the name (not the full name \
  if it's long).

CODE RULES (only if the question IS about the data):
- Store the final answer in a variable called `result`.
- Available variables:
  - `df` (main ENTRY sheet DataFrame — production data)
  - `master_packing` (MASTER PACKING sheet — packaging specs)
  - `pd` (pandas), `np` (numpy)
  - `datetime`, `timedelta` (for date calculations)
  - All sheets are also available by their clean lowercase name (spaces → underscores).
- Do NOT use import statements. Do NOT use open(), os, sys, or any I/O.
- Do NOT use exec() or eval().
- Handle potential errors (e.g., missing columns) gracefully.
- The data has BOTH numeric and text columns. You can answer:
  - MATH questions: counts, totals, averages, min/max, group-by, etc.
  - TEXT questions: "show remarks for batch X", "what stages does product Y \
    have?", "list all batches with status HOLD", "what country is product Z \
    for?" — just filter and return the relevant columns as a DataFrame.
  - MIXED questions: "which product has the highest variance?" — compute + text.
  - TEMPORAL / ORDERING questions: "last batch", "latest entry", "most recent", \
    "first batch", "oldest" — sort by the DATE column (it is datetime type) \
    and use .iloc[-1] for last or .iloc[0] for first. For date-relative queries \
    like "last 7 days", use: df[df['DATE'] >= df['DATE'].max() - timedelta(days=7)]. \
    Example for "last batch":
    last_row = df.sort_values('DATE').iloc[-1]
    result = f"Last batch: {{last_row['BATCH NO']}} — Status: {{last_row['STATUS']}}"
  - COMPARISON questions: "compare X vs Y" — group by the relevant column \
    and aggregate. Example:
    result = df[df['PRODUCT'].str.contains('BOLNOL|VRAGGRIPP', case=False, na=False)] \
      .groupby('PRODUCT').agg({{'ACTUAL QTY': 'sum', 'VARIANCE': 'sum'}})
  - PERCENTAGE questions: "what % of batches are completed?" — compute count \
    of matching rows divided by total, multiply by 100. Example:
    total = len(df)
    completed = len(df[df['STATUS'].str.contains('COMPLETED', case=False, na=False)])
    result = f"{{completed}} out of {{total}} batches are completed ({{completed/total*100:.1f}}%)"
  - TREND / PER-MONTH questions: "which month had highest production?" — \
    group by MONTH column. Example:
    result = df.groupby('MONTH')['ACTUAL QTY'].sum().sort_values(ascending=False)
  - TOP-N questions: "top 5 products by variance" — sort and head. Example:
    result = df.groupby('PRODUCT')['VARIANCE'].sum().sort_values(ascending=False).head(5)
  - PACKING / PACKAGING questions: use the `master_packing` DataFrame. Example:
    result = master_packing[master_packing['PRODUCT NAME'].str.contains('BOLNOL', case=False, na=False)]
- If the question asks for a count, total, average, etc., `result` should be \
  a number or a simple string.
- If the question asks for a list, details, or table, `result` should be a DataFrame.
- Keep code simple and concise. No plots.
- Column names are case-sensitive; use them exactly as shown in the schema.
- Return ONLY the Python code. No explanations, no markdown fences.

SPREADSHEET SCHEMA:
{schema}
"""

NOT_RELEVANT_REPLY = (
    "Sorry, I can only answer questions about the production spreadsheet data. "
    "Please ask something related to the data like batch counts, variance, "
    "product status, manpower, overtime, etc."
)

FORMAT_SYSTEM = """\
You are a helpful assistant that takes raw data results and formats them \
into a clear, concise plain-text answer for the user. \
RULES:
- ONLY use the raw data result provided. Do NOT add any outside knowledge.
- Do not use markdown formatting like asterisks or backticks.
- Be conversational and helpful.
- If the result is a table, present it neatly.
- Keep it brief and factual.
- If the data seems empty or has no results, say so clearly.\
"""


def call_llm(prompt: str, max_retries: int = 3, retry_wait: int = 60, max_tokens: int = 2048) -> str | None:
    """Call OpenRouter LLM with auto-retry on rate-limit (429) errors."""
    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            logger.error(f"LLM error (attempt {attempt + 1}): {error_str}")

            if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    logger.info(f"Rate limited — waiting {retry_wait}s before retry...")
                    time.sleep(retry_wait)
                    continue
                return None  # Exhausted retries
            # 402 = out of credits — no point retrying
            if "402" in error_str or "Payment Required" in error_str:
                logger.error("Out of OpenRouter credits — failing immediately.")
                return None
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return None
    return None


def extract_code(raw_response: str) -> str:
    """Strip markdown code fences if the model wraps its output."""
    raw = raw_response.strip()
    # Remove ```python ... ``` wrapping
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Drop first line (```python) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)
    return raw.strip()


# ── MAIN QUESTION PIPELINE ───────────────────────────────────────────────
def ask_llm(question: str, user_id: int) -> str:
    """Full pipeline: cache check → code gen → safe exec → format answer."""

    # 1. Check cache
    cached = cache.get(question)
    if cached:
        return cached

    # 2. Build conversation context (just last few Q&As, very compact)
    history = user_histories.get(user_id, [])
    history_text = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in history[-3:]]
    )

    # 3. Auto-correct typos/misspellings in the question
    corrected_question, corrections = fuzzy_correct_question(question)
    if corrections:
        logger.info(f"Fuzzy corrections: {corrections}")

    # 4. Generate pandas code
    code_prompt = CODE_GEN_SYSTEM.format(schema=SCHEMA_PROMPT)
    if history_text:
        code_prompt += f"\n\nPrevious conversation:\n{history_text}"
    code_prompt += f"\n\nUSER QUESTION: {corrected_question}\n\nPYTHON CODE:"

    raw_code = call_llm(code_prompt)
    if raw_code is None:
        return (
            "The AI model's rate limit was reached. Please wait about a "
            "minute and try again. If this persists, the daily quota may "
            "have been hit (resets at midnight Pacific time)."
        )

    code = extract_code(raw_code)
    logger.info(f"Generated code:\n{code}")

    # Check if the model flagged the question as not relevant to the data
    if "NOT_RELEVANT" in code.upper().replace(" ", "") and len(code.strip()) < 50:
        return NOT_RELEVANT_REPLY

    # 4. Execute the code safely
    success, result = execute_safely(code, DATAFRAMES)

    if not success:
        logger.warning(f"Code execution failed: {result}")
        # Retry once with the error message fed back
        retry_prompt = (
            code_prompt
            + f"\n\nThe previous code failed with: {result}\n"
            "Please fix the code and try again. Return ONLY the corrected Python code."
        )
        raw_code_2 = call_llm(retry_prompt)
        if raw_code_2:
            code_2 = extract_code(raw_code_2)
            logger.info(f"Retry code:\n{code_2}")
            success, result = execute_safely(code_2, DATAFRAMES)

        if not success:
            logger.warning(f"Retry also failed: {result}")
            # Fallback: send a small data subset as plain text
            return _fallback_answer(question, user_id)

    # 5. Format the raw result into a natural-language answer
    format_prompt = (
        FORMAT_SYSTEM
        + f"\n\nUser asked: {question}"
        + f"\n\nRaw data result:\n{result}"
        + "\n\nPlease write a clear, helpful answer:"
    )

    formatted = call_llm(format_prompt)
    if formatted is None:
        # If formatting call fails, return the raw result — still useful
        formatted = f"Here's what I found:\n\n{result}"

    # 6. Cache and return
    cache.put(question, formatted)
    return formatted


def _fallback_answer(question: str, user_id: int) -> str:
    """Last resort: send first 30 rows as text to Gemini (still much less
    than the original 122K chars approach)."""
    logger.info("Using fallback: small data subset")
    main_sheet = list(DATAFRAMES.keys())[0]
    subset = DATAFRAMES[main_sheet].head(30).to_string(index=False)

    fallback_prompt = (
        f"You are a helpful data assistant. Answer the question using ONLY "
        f"this data. If you can't answer fully, say so.\n\n"
        f"DATA (first 30 rows of {len(DATAFRAMES[main_sheet])} total):\n"
        f"{subset}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER (plain text, no markdown):"
    )

    answer = call_llm(fallback_prompt)
    if answer:
        return answer
    return (
        "Sorry, I couldn't process your question right now. "
        "Please try again in a minute."
    )


# ── TELEGRAM HANDLERS ────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Build a quick summary of what data is loaded
    sheets_info = []
    for name, df in DATAFRAMES.items():
        sheets_info.append(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
    sheets_summary = "\n".join(sheets_info)

    await update.message.reply_text(
        "Hi! I can answer questions about your production spreadsheet.\n\n"
        f"Loaded data:\n{sheets_summary}\n\n"
        "Just type a question, e.g.:\n"
        "- How many batches were COMPLETED in April?\n"
        "- What's the total variance for BOLNOL TABLET?\n"
        "- Which entries have status RUNNING?\n"
        "- Average manpower per month?\n\n"
        "Commands:\n"
        "/clear - Wipe my short-term memory\n"
        "/reload - Refresh data if the Excel file was updated\n"
        "/schema - See what columns are available"
    )


async def clear_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_histories[user_id] = []
    cache.clear()
    await update.message.reply_text("Memory and cache cleared. Starting fresh.")


async def reload_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global DATAFRAMES, SCHEMA_PROMPT
    try:
        DATAFRAMES = load_data(EXCEL_FILE_PATH)
        SCHEMA_PROMPT = get_schema_prompt(DATAFRAMES)
        cache.clear()
        sheets_info = ", ".join(
            f"{name} ({len(df)} rows)" for name, df in DATAFRAMES.items()
        )
        await update.message.reply_text(f"Data reloaded: {sheets_info}")
    except Exception as e:
        await update.message.reply_text(f"Failed to reload: {e}")


async def show_schema(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available columns and data types."""
    parts = []
    for name, df in DATAFRAMES.items():
        parts.append(f"Sheet: {name}")
        for col in df.columns:
            parts.append(f"  - {col} ({df[col].dtype})")
        parts.append("")

    text = "\n".join(parts)
    # Trim if too long for Telegram
    if len(text) > 4000:
        text = text[:4000] + "\n... (truncated)"
    await update.message.reply_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    question = update.message.text.strip()

    # Rate limit check
    if not rate_limiter.is_allowed(user_id):
        wait = rate_limiter.seconds_until_next(user_id)
        await update.message.reply_text(
            f"You're sending questions too fast. "
            f"Please wait {wait} seconds before trying again."
        )
        return

    await update.message.chat.send_action("typing")

    answer = ask_llm(question, user_id)

    # Save to conversation history (skip error messages)
    if not answer.startswith(("The AI model's rate limit", "Sorry, I couldn't")):
        user_histories.setdefault(user_id, []).append((question, answer))
        if len(user_histories[user_id]) > 4:
            user_histories[user_id].pop(0)

    # Telegram has a 4096 char limit — split long replies
    for i in range(0, len(answer), 4000):
        await update.message.reply_text(answer[i : i + 4000])


# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    try:
        from keep_alive import keep_alive

        keep_alive()
    except ImportError:
        logger.info(
            "keep_alive.py not found — skipping "
            "(fine if running as a Background Worker or locally)."
        )

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reload", reload_data))
    app.add_handler(CommandHandler("clear", clear_memory))
    app.add_handler(CommandHandler("schema", show_schema))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register an error handler to suppress transient Conflict errors during
    # Render redeploys (old + new instance briefly overlap).
    async def error_handler(update, context):
        import telegram.error
        if isinstance(context.error, telegram.error.Conflict):
            logger.warning("Conflict error (another instance running) — will resolve shortly.")
            return  # suppress; the old instance will die soon
        logger.error(f"Unhandled exception: {context.error}")

    app.add_error_handler(error_handler)

    logger.info("Bot starting (Text-to-Pandas architecture)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()