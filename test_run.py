import os
import sys
from pathlib import Path

# Force UTF-8 output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))

from agent.graph import run_agent

print("=== CHAY THU AGENT ===")
query = "Toi muon di Da Nang cuoi tuan nay tu TP.HCM, budget 5 trieu cho 2 dem, uu tien gan bien va co an sang."
print(f"Query: {query}")

try:
    result = run_agent(query, today="2026-05-31")
    print("\n--- KET QUA HOAN THANH ---")
    print(f"Final Answer:\n{result.final_answer}")
    print("\nChi tiet cac cong cu da goi (Tool calls):")
    for idx, tc in enumerate(result.tool_calls):
        print(f"[{idx+1}] Tool: {tc.name}")
        print(f"    Args: {tc.args}")
        print(f"    Output: {tc.output}")
except Exception as e:
    print("LOI KHI CHAY AGENT:", e)
