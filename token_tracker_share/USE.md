# Token Tracker — how to add LLM token/cost tracking to your project

A single-file, dependency-free helper that counts the tokens your Gemini calls use and writes a
copy-paste CSV row for a shared usage sheet. Works in **any** project layout — you only need two
files: `token_tracker.py` (the tool) and this guide.

You do **not** need to already have a CSV file. The tracker **creates one for you** the first time
it writes (see "What if I don't have a CSV file?" below).

---

## 1. What this does

- Reads the **exact token counts the provider reports** after each LLM call (Gemini counts them on
  its servers — that's what it bills on; the tracker just reads them out of the response).
- Sums them per run into a fixed-format row:

  | Use Case | Model | Iteration ID | Prompt Tokens | Cached Tokens | Output Tokens | Thinking Tokens | Total Tokens |

- Appends that row to a CSV so everyone's runs land in one sheet.
- If the provider returns no usage info, it falls back to an estimate (`len(text) // 4`,
  ~4 chars/token) and marks the row estimated — so it never produces a blank.

---

## 2. Install

Copy **`token_tracker.py`** into your project. No `pip install` needed — it uses only the Python
standard library. Put it anywhere, e.g.:

```
your_project/
├── main.py
├── utils/
│   └── token_tracker.py     <- here
└── ...
```

Import it:

```python
from utils.token_tracker import TokenTracker     # if in utils/
# or
from token_tracker import TokenTracker           # if next to your script
```

---

## 3. Use it — 3 steps

> Common ground for every project: **you already call Gemini with an API key.** That's all this
> needs. Your files, folders, and function names can be completely different from anyone else's.

### Step 1 — create ONE tracker at the start of your run
```python
from token_tracker import TokenTracker

tracker = TokenTracker(
    use_case="Your use case name",       # free text, e.g. "Churn model"
    model="gemini-2.5-flash-lite",       # the model id you call
)
```

### Step 2 — after EVERY model call, add one line
```python
from langchain_core.messages import HumanMessage

msg = llm.invoke([HumanMessage(content=prompt)])      # your existing call
tracker.record(msg, prompt=prompt, resp=msg.content, label="research")   # <-- add this
```
`label` is just a name for that step (optional). The rule: **every `invoke()` gets one `record()`
right after it** — if you forget it, those tokens won't be counted.

### Step 3 — at the end of the run, print + save
```python
tracker.print_summary()          # shows the table in your console
tracker.to_csv("cost_log.csv")   # creates the file if needed, appends one row
```

That's it. Open `cost_log.csv` and copy the row into the shared sheet.

---

## 4. What if I DON'T have a CSV file? (important)

**You don't need one.** `tracker.to_csv("cost_log.csv")` handles it automatically:

- **File doesn't exist yet** → it **creates** `cost_log.csv`, writes the **header row** first, then
  your data row.
- **File already exists** → it **appends** your data row (no duplicate header).
- **Path has folders that don't exist** (e.g. `logs/cost_log.csv`) → it creates the folders too.

So the very first run makes the file; every later run adds a line. Each teammate can point at the
**same shared CSV path** (e.g. a synced/network folder) so all rows collect in one place:

```python
tracker.to_csv(r"\\shared-drive\genai\cost_log.csv")   # everyone writes to the same file
```

If you'd rather not write a file at all, just call `tracker.print_summary()` and copy the printed
numbers manually — the CSV step is optional.

---

## 5. Different project, different files — patterns

Your LLM calls are often spread across multiple files. The trick is that **all calls must share the
same tracker object.** Two clean ways:

### Pattern A — pass the tracker into your functions (simplest)
```python
# main.py
tracker = TokenTracker(use_case="Churn model", model="gemini-2.5-flash-lite")
do_research(tracker)        # pass the same tracker everywhere
do_summary(tracker)
tracker.print_summary(); tracker.to_csv("cost_log.csv")
```
```python
# anywhere_else.py
def do_research(tracker):
    msg = llm.invoke([HumanMessage(content=prompt)])
    tracker.record(msg, prompt=prompt, resp=msg.content, label="research")
```

### Pattern B — one shared instance (if passing args is annoying)
```python
# tracker_instance.py
from token_tracker import TokenTracker
tracker = TokenTracker(use_case="Churn model", model="gemini-2.5-flash-lite")
```
```python
# any file
from tracker_instance import tracker
tracker.record(msg, prompt=prompt, resp=msg.content, label="research")
```

---

## 6. Not using LangChain? (raw SDK or other providers)

`tracker.record(msg, ...)` already understands both the **LangChain** `AIMessage` and the **raw
`google-generativeai`** response object. If you use something else, or only have the numbers, record
them explicitly:

```python
tracker.record_tokens(
    input_tokens=1500,
    output_tokens=300,
    cached_tokens=0,      # optional
    thinking_tokens=0,    # optional
    label="research",
)
```

---

## 7. About Cached & Thinking tokens (so you don't think they're broken)

These are **often 0, and that's correct**:

- **Thinking tokens** — only Gemini 2.5 *thinking/reasoning* models emit these. On
  `gemini-2.5-flash-lite` thinking is off by default → 0. Switch to `gemini-2.5-flash` / `-pro`
  with thinking enabled to see non-zero values.
- **Cached tokens** — this counts Gemini's *server-side context cache*. Most projects don't enable
  it, so it stays 0. (If your project keeps its own local response cache, that's separate and also
  shows 0 here.)

The **Prompt** and **Output** token counts are the real, provider-reported numbers — those are what
matter for usage tracking.

---

## 8. Optional: estimated $ cost

`print_summary()` also shows an estimated USD cost using a built-in pricing table (placeholders —
**verify against current Google pricing**). Override it if you know your rate:

```python
tracker = TokenTracker(use_case="...", model="...",
                       pricing={"input": 0.00010, "output": 0.00040})   # USD per 1,000 tokens
# or set env vars: GEMINI_PRICE_INPUT_PER_1K / GEMINI_PRICE_OUTPUT_PER_1K
```

The CSV itself contains only token counts, not cost — so the shared sheet stays provider-agnostic.

---

## 9. Full minimal example

```python
from token_tracker import TokenTracker
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os

os.environ["GOOGLE_API_KEY"] = "AIza..."          # your key (you already have this)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

tracker = TokenTracker(use_case="Demo", model="gemini-2.5-flash-lite")

for step, prompt in [("research", "Summarise X"), ("design", "Propose Y")]:
    msg = llm.invoke([HumanMessage(content=prompt)])
    tracker.record(msg, prompt=prompt, resp=msg.content, label=step)

tracker.print_summary()
tracker.to_csv("cost_log.csv")        # cost_log.csv now exists with header + 1 row
```
