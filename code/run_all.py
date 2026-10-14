"""run_all.py -- run every data-type demo in one go: generate sample data and figures."""  # module comment
import runpy                                                          # use runpy to execute the other .py files as scripts
import sys                                                            # used to modify the module search path
import time                                                           # used to time each script
from pathlib import Path                                              # path utility

HERE = Path(__file__).resolve().parent                                # folder of this script: website/code
sys.path.insert(0, str(HERE))                                         # ensure sub-scripts can import _common
SCRIPTS = ["01_datatype_numeric.py",                                  # numeric (continuous / discrete)
           "02_datatype_categorical.py",                              # categorical (nominal / ordinal)
           "03_datatype_text.py",                                     # text
           "04_datatype_image.py",                                    # image
           "05_datatype_audio.py",                                    # audio
           "06_datatype_video.py",                                    # video
           "07_datatype_graph.py",                                    # graph / network
           "08_datatype_spatiotemporal.py",                           # spatiotemporal
           "09_datastructure_cross_time_panel.py",                    # cross-section / time series / panel
           "10_datatype_structured_unstructured.py"]                  # structured / semi-structured / unstructured
for name in SCRIPTS:                                                  # run each script in turn (order matches the lecture)
    t0 = time.time()                                                  # record the start time
    print("\n===== run", name, "=====")                               # print the current script name to track progress
    runpy.run_path(str(HERE / name), run_name="__main__")             # run the script as if it were the main program
    print("----- done %s in %.2fs" % (name, time.time() - t0))        # print the elapsed time
print("\nAll done: samples in website/data/, figures in website/figures/")  # final note on the output locations
