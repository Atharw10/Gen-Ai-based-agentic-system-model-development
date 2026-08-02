"""Human-in-the-loop gate - synchronous input() with safe parsing."""
from __future__ import annotations
from typing import import Dict, Any
from IPython.display import import display, Markdown


def hitl_gate(
    stage_name: str,
    summary: str,
    proposed_decisions: Dict[str, Any],
    options: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Present results to user and collect override instructions.
    
    Args:
        stage_name: e.g., 'cleaning', 'model_design'
        summary: short markdown describing what was decided
        proposed_decisions: dict of {decision_name: current_value}
        options: dict of {decision_name: 'comma-separated alternatives'}
        
    Returns:
        dict with keys:
            action       : 'accept' | 'override' | 'regenerate'
            overrides    : dict of {decision_name: new_value} (if override)
            instructions : str (if regenerate - free-form for LLM)
    """
    display(Markdown(f"## 🧑‍💻 HITL Gate - '{stage_name}'"))
    display(Markdown(summary))
    display(Markdown("### Proposed decisions"))
    for k, v in proposed_decisions.items():
        opts = f" _(options: {options[k]})_" if options and k in options else ""
        display(Markdown(f"- **{k}** = `{v}`{opts}"))
        
    print("\n" + "-" * 50)
    print(f"Choose your action for stage [{stage_name}]:")
    print("[a] Accept proposed decisions and continue")
    print("[o] Override specific decisions")
    print("[r] Regenerate the whole stage with free-form instructions")
    print("-" * 50)
    
    while True:
        choice = input("Your choice [a/o/r]: ").strip().lower() or "a"
        if choice in ("a", "o", "r"):
            break
        print("Invalid input, choose a / o / r")
        
    if choice == "a":
        print(f"✅ Accepted all proposed decisions for [{stage_name}]")
        return {"action": "accept", "overrides": {}, "instructions": ""}
        
    if choice == "o":
        overrides = {}
        print("\nFor each decision, type a new value or press Enter to keep current.")
        for k, v in proposed_decisions.items():
            opts = f" (options: {options[k]})" if options and k in options else ""
            new = input(f"  {k} [{v}]{opts}: ").strip()
            if new:
                overrides[k] = _coerce(new, v)
        print(f"🛠️ Overrides recorded: {overrides}")
        return {"action": "override", "overrides": overrides, "instructions": ""}
        
    # choice == "r"
    print("\nProvide free-form instructions for regenerating this stage:")
    print("(e.g., 'use mean instead of median, also drop columns with >30% nulls')")
    instructions = input("Instructions: ").strip()
    print(f"🔄 Regeneration requested with instructions")
    return {"action": "regenerate", "overrides": {}, "instructions": instructions}


def _coerce(s: str, ref):
    """Try to convert user input to the type of the reference value."""
    if isinstance(ref, bool):
        return s.lower() in ("y", "yes", "true", "1")
    if isinstance(ref, int):
        try:
            return int(s)
        except Exception:
            return s
    if isinstance(ref, float):
        try:
            return float(s)
        except Exception:
            return s
    if isinstance(ref, list):
        return [x.strip() for x in s.split(",") if x.strip()]
    return s