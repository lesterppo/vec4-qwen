"""Deterministic instruction dispatcher for VEC4 tools (v3, no model needed).

v3 extends v2 with VSHL/VSHR/MMUL/VSORT tools and semantic patterns.
The contract is unchanged: semantic task beats mentions, any
disagreeing mention flags CONFLICT, ambiguous -> refusal. For the
new ops: shift amounts come from the B vector's lanes (a request
"shift left by the amounts in [...]" is executed with those lanes
as B); MMUL is a 2x2 matrix product; VSORT ignores B (its values
do not affect the output). Exception to the v2 vector rules, added
2026-11-05 after the 2B-v3 gate: a VSORT request with a single
vector executes with a dummy B of zeros, which VSORT never reads.
Every other op still requires two vectors.


This is the answer to the tool-selection failure, generalised: the
model must never be the thing that decides which chip tool runs.
The dispatcher decides, from the USER's own text, and the model is
only ever asked to verbalise a result the dispatcher already computed.

Rules, in order (this order is the whole design):

1. Semantic task wins over tool-name mentions. "Use vec_add to
   compute the dot product" is a DOT request with a CONFLICT, not
   an ADD request: the task noun phrase ("dot product") is the
   intent, the tool name is a mention that can be injected or
   misdirected. The dispatcher records the conflict, executes DOT,
   and never executes the mentioned wrong tool.
2. A single tool mention with NO semantic signal is accepted as
   the intent ("Use vec_dot." with two vectors). That is how the
   original tool tests phrased requests, and refusing it would
   break the normal case. Multiple conflicting mentions with no
   semantic signal are AMBIGUOUS, never guessed.
3. Vectors are parsed from the user's text (two bracket lists of
   4 ints, each 0..15). No vectors, or out-of-range values, is an
   error, never a silent default. Exception: VSORT accepts a single
   list (its B is ignored, so a dummy B of zeros is supplied).
   A list marked forget/ignore is
   a distractor and is excluded; more than two remaining lists
   is AMBIGUOUS (refusal, not a guess). "Do not compute X;
   instead Y" takes its op from the clause after "instead".
   "Subtract X from Y" / "Take X away from Y" swap operands.
4. For programs (multi-step), the dispatcher guarantees EXECUTION
   fidelity only: it executes exactly the steps given, in order,
   each step verified against the reference, with $n refs to
   earlier step results. It does NOT guarantee that a model-
   invented program matches the user's intent — intent fidelity
   for programs is only claimed when the program itself comes
   from the user's request. A model that proposes a program is
   proposing, not deciding: the dispatcher (or the user) must
   approve the program before execution. This boundary is
   deliberate, and collapsing it is how tool-selection failures
   come back.

The executor is injected (in tests: ref.lane_result / dot sum).
In a Colab run it is the chip: encode(op, A, B) -> run merged
model -> decode OUT tokens -> result. The dispatcher's contract
does not change when the executor changes.
"""

import re

TOOLS = {"vec_add": "ADD", "vec_dot": "DOT", "vec_sub": "SUB",
         "vec_mul": "MUL", "vec_max": "MAX", "vec_min": "MIN",
         "vec_vshl": "VSHL", "vec_vshr": "VSHR", "vec_mmul": "MMUL",
         "vec_vsort": "VSORT"}

# Semantic patterns, checked in order: dot product must precede any
# generic "product/multiply" pattern, or DOT requests become MUL;
# "larger"/"smaller" must precede generic maximum/minimum, or a
# sentence mentioning both ("I mix up minimum and maximum; give the
# smaller") resolves to the wrong one. v3: the four new-op patterns
# come FIRST, before every pattern they could collide with:
# "matrix product" would match generic "product" -> MUL, and
# "smallest to largest" would match "smallest" -> MIN.
# 2026-11-05 gate fixes: "product of the two matrices" is MMUL, not
# generic "product" -> MUL, so the MMUL pattern also matches
# "product ... matrices"; "in ascending order" is VSORT, not
# "smallest" -> MIN, so the VSORT pattern also matches it; "to the
# left" / "to the right" are shift directions even without "shift"
# adjacent to the direction word.
SEMANTIC = [
    (r"matrix product|multiply .* matrices|matrix multipl|product .* matrices", "MMUL"),
    (r"shift .* left by|shift left|to the left", "VSHL"),
    (r"shift .* right by|shift right|to the right", "VSHR"),
    (r"sort .* ascending|sort .* smallest to largest|sort ascending|in ascending order|ascending order", "VSORT"),
    (r"dot product|scalar product|sum of (?:the )?(?:pairwise )?products|dotted\b|\bdot\b", "DOT"),
    (r"lane[- ]by[- ]lane (?:maximum|greatest|largest)|element[- ]wise max", "MAX"),
    (r"lane[- ]by[- ]lane (?:minimum|smallest)|element[- ]wise min", "MIN"),
    (r"lane[- ]by[- ]lane (?:multipl|product)|multiply .* lane|corresponding .* multipl", "MUL"),
    (r"lane[- ]by[- ]lane (?:subtract|differenc)|subtract .* lane", "SUB"),
    (r"lane[- ]by[- ]lane (?:add|sum)|add .* lane", "ADD"),
    (r"take .* away from", "SUB"),
    (r"element[- ]?wise product", "MUL"),
    (r"\bsmaller\b", "MIN"),
    (r"\blarger\b", "MAX"),
    (r"\bmaximum\b|\bgreatest\b|\blargest\b", "MAX"),
    (r"\bminimum\b|\bsmallest\b", "MIN"),
    (r"\bminus\b", "SUB"),
    (r"\bsubtract", "SUB"),
    (r"\bmultipl", "MUL"),
    (r"\bproduct\b", "MUL"),
    (r"\bplus\b", "ADD"),
    (r"\badd\w*|\bsum\b", "ADD"),
]

VEC_RE = re.compile(r"\[([0-9,\s]+)\]")


def semantic_op(text):
    """First matching semantic op, or None. Order matters (see above).

    "Do not compute X; instead Y" construction: the operative task
    is the clause after "instead", not the negated one. Search that
    clause first; fall back to the full text only if it has no op.
    """
    low = text.lower()
    if "instead" in low:
        clause = low.split("instead", 1)[1]
        for pat, op in SEMANTIC:
            if re.search(pat, clause):
                return op
    for pat, op in SEMANTIC:
        if re.search(pat, low):
            return op
    return None


def mentioned_tools(text):
    """Tool names mentioned in the text, in order, deduplicated."""
    out = []
    for m in re.finditer(r"vec_(add|dot|sub|mul|max|min|vshl|vshr|mmul|vsort)\b", text.lower()):
        name = "vec_" + m.group(1)
        if name not in out:
            out.append(name)
    return out


def _scan_vectors(text):
    """Operative bracket lists of 4 ints, after forget-marker and
    distractor filtering. No count or range checks here; callers
    apply their own. Shared by parse_vectors and the VSORT
    single-vector path so both see exactly the same lists."""
    m_marker = re.search(r"forget that|actually,? forget", text.lower())
    search_text = text[m_marker.end():] if m_marker else text
    vecs = []
    for m in VEC_RE.finditer(search_text):
        prefix = search_text[max(0, m.start() - 20):m.start()].lower()
        if "forget" in prefix or "ignore" in prefix:
            continue
        try:
            vals = [int(x) for x in m.group(1).split(",") if x.strip() != ""]
        except ValueError:
            continue
        if len(vals) == 4:
            vecs.append(vals)
    return vecs


def parse_vectors(text):
    """Two bracket lists of 4 ints each, or (None, error).

    A bracket list explicitly marked forget/ignore in the preceding
    words is a distractor and is excluded. A "forget that" /
    "actually, forget" marker abandons the earlier task, so only
    lists after that marker are operative. More than two remaining
    lists (e.g. a claimed answer quoted in the prompt) is
    TOO_MANY_VECTORS: the dispatcher refuses rather than guessing
    which two are the operands. Refusal is a safe outcome, not a
    wrong computation, and is recorded as AMBIGUOUS by dispatch().
    """
    vecs = _scan_vectors(text)
    if len(vecs) < 2:
        return None, "NO_VECTORS"
    if len(vecs) > 2:
        return None, "TOO_MANY_VECTORS"
    A, B = vecs[0], vecs[1]
    if not all(0 <= v <= 15 for v in A + B):
        return None, "OUT_OF_RANGE"
    return (A, B), None


def parse_single_vector(text):
    """Exactly one operative 4-int list, or (None, error).

    Only used for VSORT, whose B operand is ignored: a single list
    executes with a dummy B. Zero lists is NO_VECTORS (no intent
    signal to execute); an out-of-range list is OUT_OF_RANGE, never
    silently clipped; more than one list is not this function's
    case (the two-vector path owns it).
    """
    vecs = _scan_vectors(text)
    if len(vecs) != 1:
        return None, "NO_VECTORS"
    if not all(0 <= v <= 15 for v in vecs[0]):
        return None, "OUT_OF_RANGE"
    return vecs[0], None


def dispatch(text):
    """Decide the intended op from the user's text. No execution here.

    Returns a dict with status: OK (semantic or single mention),
    CONFLICT (semantic beats a disagreeing mention), AMBIGUOUS
    (no semantic signal and 0 or 2+ distinct mentions), or an
    error status (NO_VECTORS / OUT_OF_RANGE).
    """
    sem = semantic_op(text)
    mentions = mentioned_tools(text)
    mentioned_ops = [TOOLS[m] for m in mentions]
    vecs, err = parse_vectors(text)
    # VSORT single-vector path (2026-11-05 gate fix): VSORT ignores
    # B, so one list is enough -- but only when the intended op
    # actually resolves to VSORT (semantic or single mention). A
    # bare list with no op signal still refuses, and every other
    # op still requires two vectors.
    if err == "NO_VECTORS":
        intended = sem if sem is not None else (
            mentioned_ops[0] if len(mentioned_ops) == 1 else None)
        if intended == "VSORT":
            single, single_err = parse_single_vector(text)
            if single is not None:
                vecs, err = (single, [0, 0, 0, 0]), None
            elif single_err is not None:
                err = single_err
    if err:
        if err == "TOO_MANY_VECTORS":
            return {"status": "AMBIGUOUS", "semantic_op": sem, "mentioned": mentions,
                    "intended_op": None, "vectors": None}
        return {"status": err, "semantic_op": sem, "mentioned": mentions,
                "intended_op": None, "vectors": None}
    # "Subtract X from Y" / "Take X away from Y" mean Y - X, but
    # vectors arrive in textual order (X, Y). Swap for SUB when
    # either construction is used; "subtract A and B" keeps textual
    # order (A - B).
    if sem == "SUB" and re.search(r"(subtract|take .* away).*from\b", text.lower()):
        vecs = (vecs[1], vecs[0])
    if sem is not None:
        status = "OK"
        if any(op != sem for op in mentioned_ops):
            status = "CONFLICT"
        return {"status": status, "semantic_op": sem, "mentioned": mentions,
                "intended_op": sem, "vectors": vecs}
    if len(mentioned_ops) == 1:
        return {"status": "OK", "semantic_op": None, "mentioned": mentions,
                "intended_op": mentioned_ops[0], "vectors": vecs}
    return {"status": "AMBIGUOUS", "semantic_op": None, "mentioned": mentions,
            "intended_op": None, "vectors": vecs}


def execute(text, executor):
    """Dispatch + execute a single-op request. executor(op, A, B) -> result.

    Result is a lane list for lane ops, or an int for DOT. Only
    OK / CONFLICT statuses execute; AMBIGUOUS and errors do not.
    """
    d = dispatch(text)
    if d["status"] not in ("OK", "CONFLICT"):
        return {**d, "result": None}
    A, B = d["vectors"]
    result = executor(d["intended_op"], A, B)
    return {**d, "result": result}


def execute_program(steps, executor):
    """Execute a program: list of {op, a, b} where a/b are literal
    4-lists or "$n" refs to an earlier step's result (lane list for
    lane ops; for DOT, a step result is an int and cannot be a
    vector operand — that is a PROGRAM_ERROR, not a silent cast).

    Returns per-step results and a transcript. Any invalid step
    aborts the whole program with PROGRAM_ERROR; no partial result
    is presented as a program result.
    """
    results = []
    transcript = []
    for i, step in enumerate(steps):
        op = step.get("op")
        if op not in ("ADD", "DOT", "SUB", "MUL", "MAX", "MIN",
                      "VSHL", "VSHR", "MMUL", "VSORT"):
            return {"status": "PROGRAM_ERROR", "step": i,
                    "error": "unknown op %r" % (op,), "results": results}
        vecs = []
        for key in ("a", "b"):
            v = step.get(key)
            if isinstance(v, str) and v.startswith("$"):
                try:
                    idx = int(v[1:])
                    v = results[idx]
                except (ValueError, IndexError):
                    return {"status": "PROGRAM_ERROR", "step": i,
                            "error": "bad ref %r" % (v,), "results": results}
            if not (isinstance(v, list) and len(v) == 4
                    and all(isinstance(x, int) and 0 <= x <= 15 for x in v)):
                return {"status": "PROGRAM_ERROR", "step": i,
                        "error": "bad operand %s=%r" % (key, v), "results": results}
            vecs.append(v)
        res = executor(op, vecs[0], vecs[1])
        results.append(res)
        transcript.append({"step": i, "op": op, "a": vecs[0], "b": vecs[1],
                           "result": res})
    return {"status": "OK", "results": results, "transcript": transcript}
