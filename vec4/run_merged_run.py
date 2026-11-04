import sys
sys.argv = ["run_merged.py", "/content/qwen35-08b-vec4", "/content/qwen35-08b"]
sys.path.insert(0, "/content/qwenputer-colab/vec4")
import run_merged
run_merged.main()
