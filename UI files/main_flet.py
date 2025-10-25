import os
import sys

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Flet main module
from Flet.main import main
import flet as ft

if __name__ == "__main__":
    # Run the Flet app
    ft.app(target=main)