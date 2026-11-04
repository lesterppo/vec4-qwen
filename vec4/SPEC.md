# VEC4 — a tiny GPU-style vector unit compiled into Qwen3.5-0.8B weights

## What is new, and what is reused

NEW: VEC4 itself — a 4-lane SIMD/SIMT vector unit (one instruction, four
lanes in parallel), its token encoding, its reference simulator, and its
compiled circuit (lane adders, a product adder tree, register/result
DeltaNet state, and an exit code that emits one output token per step).
REUSED (unmodified, from divagr18/qwenputer, inspection clone at
`../upstream/`): the chipformer IR (`q35/ir.py`: Graph/FFN/Head/Target,
exactness evaluator), exact gadgets (`q35/prims.py`: `t_and`, `t_ge`,
`slot_head`), the weight emitter (`q35/emit.py`), the standalone
0.8B-shaped model builder (`q35/tiny.py`), and the generic widen/mute
parts of `q35/merge.py` (Stage 3 only). This is NOT the author's CHIP-8
machine, NOT the RV32 machine, and NOT QwenPU: no CHIP-8 BIOS, no RISC-V
instructions, no vector-unit code from those projects is used. The only
shared idea is the one the task itself prescribes — a compiled circuit
in Gated DeltaNet + SwiGLU layers — and the author's compiler library
that implements that idea exactly.

## Why "GPU-style"

A scalar CPU would compute lane 0, then lane 1, then lane 2, then lane 3,
in successive steps. VEC4 does the opposite: at the GO position, all
four lanes' additions AND all four lanes' products are computed in the
same FFN sublayers, in parallel, by distinct units/heads. The sequential
part is only the output: a language model can only emit one token at a
time, so the four lane results (or the three dot hex digits) are emitted
one token per generation step, read back from DeltaNet state. The
computation itself is parallel; the serialisation is in the token
stream, exactly as in the author's machines (their trace is also serial),
but here the parallel lanes are the point of the design, not an
afterthought.

## ISA

- 4 lanes, lane `i` in 0..3.
- Each lane value is an unsigned 4-bit integer, 0..15.
- Vector registers: A = (A0,A1,A2,A3), B = (B0,B1,B2,B3). Registers are
  NOT read from token features at the compute position: they are written
  into DeltaNet state, one head per register, when their input token
  arrives, and read back at GO. Why state, not token features? At GO,
  the residual holds only the GO token's features; the eight input
  values arrived at eight earlier positions and are otherwise gone.
  DeltaNet state is the only exact, per-position memory the architecture
  offers, and it is precisely the mechanism the task prescribes. A
  token-feature-only design would need the values to still be present
  at GO, which they are not.
- Operations, one per program (the OP token selects it):
  - `VADD`: lane-wise addition mod 16. Result lane `i` =
    `(A[i] + B[i]) mod 16`. All four lanes computed in parallel.
  - `VDOT`: `sum_i A[i] * B[i]`. Max value 4*15*15 = 900, which fits in
    10 bits. All four products computed in parallel (a product adder
    tree), then summed.
  - `VMUL` lane-wise is NOT included: it would be a subset of the dot
    product work and adds no new gate. It is a bonus only if it costs
    no gate; it does not, so it is out of scope by design.

## Token encoding (standalone tiny-model vocab, 244 tokens, padded to 256)

Token ids are the standalone vocab ids. Stage 3 maps every one of these
onto a chip token appended after Qwen's 248320-token vocab (see
`RESULT.md` Stage 3 section when it is written); the circuit itself is
unchanged by that renumbering, because token features are functions of
the token's VEC4 kind/value, not of its absolute id — the Stage 3
driver translates ids before feeding them to the model.

| Tokens | Ids | Meaning |
|---|---|---|
| `OP_ADD` | 0 | program is VADD |
| `OP_DOT` | 1 | program is VDOT |
| `VAL r v` | 2 + r*16 + v, r in 0..7, v in 0..15 | register r holds value v. r 0..3 = A0..A3, r 4..7 = B0..B3 |
| `GO` | 130 | end of input; compute now, emit first output token next |
| `OUT_ADD l d` | 131 + l*16 + d, l in 0..3, d in 0..15 | VADD output: lane l result digit d |
| `OUT_DOT p d` | 195 + p*16 + d, p in 0..2, d in 0..15 | VDOT output: hex digit p of the dot (p 0 = high = dot>>8, 1 = mid = (dot>>4)&15, 2 = low = dot&15) |
| `END` | 243 | end of output; host stops generating here |

Input sequence (fixed order, 10 tokens):
`[OP, VAL A0, VAL A1, VAL A2, VAL A3, VAL B0, VAL B1, VAL B2, VAL B3, GO]`

Output sequence (generated, host stops at END):
- VADD: `[OUT_ADD 0 d0, OUT_ADD 1 d1, OUT_ADD 2 d2, OUT_ADD 3 d3, END]`
  where `di = (A[i]+B[i]) mod 16`.
- VDOT: `[OUT_DOT 0 hi, OUT_DOT 1 mid, OUT_DOT 2 lo, END]`
  where `dot = sum A[i]*B[i]`, `hi = dot>>8`, `mid = (dot>>4)&15`,
  `lo = dot&15`. Example: all-15 inputs give dot = 900 = 0x384, so
  `[OUT_DOT 0 3, OUT_DOT 1 8, OUT_DOT 2 4, END]`.

No other tokens are generated. At input positions the model's next-token
prediction is irrelevant: the host teacher-forces the input sequence.
At GO and at output positions the circuit owns the position (see
"Output code" below); at every other position it does not.

## Circuit sketch (full detail is in `circuit.py`, not prose)

1. Token features (level 0): `is_op`, `op_add`, `op_dot`, `is_val`,
   register bits (3), value bits (4), `is_go`, `is_out`, `is_add_out`,
   lane bits (2), `is_dot_out`, place bits (2), `is_end`, and the
   8 output-code dims (see below). Every feature is 0/1 except the
   code dims, which are `+128`/`-128` (see `q35/chip8_q35.py`'s
   `CODE_AMP` convention, reused here).
2. State heads (first head sublayer, 9 heads, parallel):
   - 8 register heads, one per register, 1 slot each. Key = "this token
     is VAL for this register", value = its 4 value bits, query =
     `is_go`. At a VAL token the head writes its row; at GO it reads
     the row back as 4 bit dims; elsewhere it outputs 0.
   - 1 OP head, 1 slot. Key = `is_op`, value = (op_add, op_dot) bits,
     query = `is_go`. At GO it outputs the op bits.
3. Compute FFNs (parallel lanes):
   - Lane adders: per lane, a 4-bit binary adder (carry chain with
     `t_ge` carry nodes and sum nodes). Result bits `s0..s3` per lane;
     the final carry is computed (needed for bit 3's sum) and discarded
     (that is the mod-16 behaviour).
   - Dot product: per lane, 16 AND nodes for the partial-product bits
     `a_i * b_j`; then an adder tree: add partial products 0+1 and
     2+3 in parallel (8/9-bit adders), then add those two sums
     (9-bit adder) to give 10 dot bits. Dot is never held as a single
     bf16 integer: 900 is not bf16-exact, so the dot lives only as its
     10 bits, produced by the adder tree. The same reason lane sums
     live as bits, not as integers above 30 (those would be exact, but
     bits are what the output needs).
   - All lanes' adders run in the same sublayers (units permitting;
     the graph placer spreads them only if a sublayer would exceed
     the 3584-unit cap). That sharing is the SIMD parallelism.
4. Result write/read heads (one head sublayer after compute, 5 heads):
   - 4 lane heads, 1 slot each: key = `is_go`, value = lane sum bits,
     query = `is_out`. At GO they write their lane's result row; at
     output positions they read it back.
   - 1 dot head, 1 slot: key = `is_go`, value = 10 dot bits, query =
     `is_out`. Same write/read pattern.
   - At GO these heads output 0 (query is 0 there); the exit code at GO
     therefore uses the compute bits directly, not the result-head
     outputs. At output positions the compute bits are 0 (register
     heads query only at GO), and the exit code uses the result-head
     outputs. The two cases are mutually exclusive by construction
     (`is_go` vs `is_out`), so one exit code serves both.
5. Output code (exit, last chip sublayer):
   - 8 code dims: flag (1 for any output token, 0 otherwise) + 7 bits
     of an output index. Output index: ADD outputs 0..63
     (`lane*16+digit`), DOT outputs 64..111 (`64+place*16+digit`),
     END 112.
   - Code dims are token features (`+128` for bit 1, `-128` for bit 0)
     on input, cleared by the exit-clear nodes, then set by exit nodes
     to `KAPPA * (2*bit-1)` (`KAPPA = 2**-20`) exactly where the circuit
     owns the position (`owner = is_go + is_out`, a 0/1 node), and to
     0 elsewhere. The tiny model's final norm (gain `2**20` on code
     dims, `-1` elsewhere, as in `tiny.build`) makes the lm_head
     argmax the token whose code matches the exit code. This is the
     author's exit mechanism, reused unchanged in form.
   - Next-token selection at each owned position, from the last
     token's type and the result bits:
     - GO + ADD -> ADD lane 0 digit (lane-0 sum bits)
     - GO + DOT -> DOT high digit (dot bits 8..9, upper bits 0)
     - ADD lane l (l<3) -> ADD lane l+1 digit (lane l+1 result bits)
     - ADD lane 3 -> END
     - DOT high -> DOT mid digit (dot bits 4..7)
     - DOT mid -> DOT low digit (dot bits 0..3)
     - DOT low -> END
   - Each code bit is a sum over the exclusive cases that set it, each
     case an AND (`t_and`) of its condition literals and its digit bit
     (or no digit bit, for bits fixed by the index, e.g. END's bits
     4,5,6 and DOT's bit 6). No case is ever active at two positions
     with different digits, because the conditions partition positions
     by last-token type.

## Exactness rules (from `ir.py`, enforced by `ir.evaluate`)

- Every FFN gate input, after scaling by K=128, must be 0, >= 18, or
  <= -104 (so silu is exactly relu / exactly 0 in bf16).
- Every head key must be one-hot (or zero); every query multi-hot
  with equal values (or zero); decay selectors only 0, 1, 0.5 in the
  encoded form; state-row data columns only 0 or `A_BIT`.
- Every materialised value must be bf16-exact. In particular, no
  intermediate holds an integer above 256 as a single value (dot = 900
  is bits-only, as above; a product = 225 is exact, a lane sum = 30
  is exact).
- `ir.evaluate(..., check=True)` is the gate: any violation raises
  `ExactnessError` naming the node. Stage 2 passes only if it raises
  for no case in the test set and the decoded next token matches the
  reference at every owned position.

## Reference simulator (`ref.py`)

Pure Python, no torch, no chipformer. It implements only the encoding
above and the arithmetic in the ISA section, so that `test_circuit.py`
can compare circuit output against it token-for-token. The simulator
is deliberately NOT shared code with the circuit: its arithmetic is
plain Python `+`, `*`, `%`, `>>`, `&`, while the circuit's is AND
gates, carry chains and adder trees. Agreement between two independent
implementations is the check.
