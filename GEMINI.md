# GEMINI.md

Behavioral guidelines to reduce common LLM coding mistakes, adapted for Gemini and Antigravity. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

---

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No defensive error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: *"Would a senior engineer say this is overcomplicated?"* If yes, simplify.

---

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, naming conventions, and patterns, even if you'd do it differently.
- If you notice unrelated dead code or bugs, mention them — don't delete or touch them unprompted.

When your changes create orphans:
- Remove imports/variables/functions that **your changes** made unused.
- Do not remove pre-existing unused code unless explicitly asked.

**The Test:** Every changed line should trace directly to the user's request.

---

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests / verify with invalid inputs, then make them pass"
- "Fix the bug" → "Reproduce the issue first, apply fix, then verify it passes"
- "Refactor X" → "Ensure functionality and tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check / test command]
2. [Step] → verify: [check / test command]
3. [Step] → verify: [check / test command]
```

Strong success criteria let the agent loop independently and safely. Weak criteria ("make it work") require constant clarification.

---

## 5. Antigravity & Tooling Hygiene

When operating in this workspace:
- **Build & Verification:** Always run `dotnet build` or relevant tests after modifying code to verify zero errors and zero warnings.
- **Background Tasks:** Do not leave dangling background processes. Terminate test servers after verification.
- **Source of Truth:** Respect `docs/schema.sql` and `ARCHITECTURE-SPINE.md` as the project's invariants.

---

**These guidelines are working if:**
- Diffs contain zero unnecessary or stylistic changes to unrelated code.
- No rewrites occur due to speculative over-engineering.
- Clarifying questions and tradeoff analysis happen **before** implementation rather than after mistakes.
