import sys
sys.path.insert(0, "/content/qwenputer-colab/vec4")
sys.path.insert(0, "/content/qwenputer-colab/upstream")
import circuit
print("circuit file:", circuit.__file__, flush=True)
import inspect
print("build_vec4 signature:", inspect.signature(circuit.build_vec4), flush=True)
c = circuit.build_vec4(offset=248320)
for name in ("mu", "is_text", "is_go", "tau", "code0"):
    for d, dim in enumerate(c.dims):
        if dim.name == name:
            print(name, "GO", c.feature_value(d, 248320 + 130),
                  "VAL0", c.feature_value(d, 248320 + 2),
                  "Qwen0", c.feature_value(d, 0), flush=True)
            break
