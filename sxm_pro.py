#!/usr/bin/env python3
"""
SatStash - The Ultimate SiriusXM Terminal App
Launch with: python sxm_pro.py
"""
import sys
from pathlib import Path

# Add sxm module to path
sys.path.insert(0, str(Path(__file__).parent))

from sxm.ui.app import SiriusXMPro

def main():
    """Main entry point"""
    app = SiriusXMPro()
    app.run()

if __name__ == "__main__":
    main()
