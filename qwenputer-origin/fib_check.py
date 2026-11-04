"""Standalone RV32 ISA free run of c-fib, vs qemu.json expected output.
Run from experimental/riscv with system python (chipformer is pure python+numpy).
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from chipformer.rv import elf, ref

cpu = ref.Rv32()
elf.load_into(cpu, Path("tests/data/rv/c-fib.qemu.elf"))
cpu.x[2] = 0x7FFF0000
steps = 0
while not cpu.halted and steps < 20_000_000:
    cpu.step()
    steps += 1
print("halted", cpu.halted, "steps", steps, "exit", cpu.exit_code,
      "stdout", repr(cpu.out.decode("latin-1")))
