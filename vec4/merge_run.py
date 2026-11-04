import sys
sys.argv = ["merge_vec4.py", "/content/qwen35-08b", "/content/qwen35-08b-vec4"]
sys.path.insert(0, "/content/qwenputer-colab/vec4")
sys.path.insert(0, "/content/qwenputer-colab/upstream")
import merge_vec4
merge_vec4.main()
