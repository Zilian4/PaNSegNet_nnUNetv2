# Add the network_architecture directory to sys.path once at package level
# This allows dynamic_network_architectures to be imported as a top-level module
import sys
import os

_file_dir = os.path.dirname(os.path.abspath(__file__))
if _file_dir not in sys.path:
    sys.path.insert(0, _file_dir)

