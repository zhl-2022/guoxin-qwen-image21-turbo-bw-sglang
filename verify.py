"""Official accuracy material gate; never fabricates a score."""
from pathlib import Path
import runpy,sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/"tools"))
sys.argv=[str(root/"tools/qwen21_standard_tools007.py"),"verify","--root",str(root),*sys.argv[1:]]
runpy.run_path(sys.argv[0],run_name="__main__")
