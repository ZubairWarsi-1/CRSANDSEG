"""
Single-command runner to generate all documentation diagrams and compile all three PDFs:
1. Customer_360_Complete_Project_Documentation.pdf
2. Customer_360_User_Manual.pdf
3. Customer_360_Technical_Documentation.pdf
"""

import subprocess
import sys
import os

DOC_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 70)
    print("CUSTOMER360 AUTOMATED DOCUMENTATION COMPILATION SUITE")
    print("=" * 70)
    
    # Step 1: Generate Diagrams
    print("\n[Step 1/2] Generating high-resolution architecture & ML diagrams...")
    gen_diag = os.path.join(DOC_DIR, "generate_diagrams.py")
    res = subprocess.run([sys.executable, gen_diag], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error generating diagrams:\n{res.stderr}")
        sys.exit(res.returncode)
    print(res.stdout.strip())
    
    # Step 2: Build All 3 PDFs
    print("\n[Step 2/2] Compiling production ReportLab PDF documents...")
    build_doc = os.path.join(DOC_DIR, "build_documentation.py")
    res = subprocess.run([sys.executable, build_doc], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error compiling PDFs:\n{res.stderr}")
        sys.exit(res.returncode)
    print(res.stdout.strip())
    
    print("\n" + "=" * 70)
    print("ALL DOCUMENTATION DELIVERABLES COMPILED AND VERIFIED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
