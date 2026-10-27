import os
import subprocess
import tempfile
import pytest

DIFF_RULES_PY = os.path.join(os.path.dirname(__file__), "..", "tools", "diff_rules.py")

def run_diff_rules(diff_content):
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".diff") as f:
        f.write(diff_content)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["python", DIFF_RULES_PY, temp_path],
            capture_output=True,
            text=True
        )
        return result.returncode, result.stdout.strip()
    finally:
        os.remove(temp_path)

def test_ok_diff():
    diff = """--- a/python/sglang/test.py
+++ b/python/sglang/test.py
@@ -1,1 +1,2 @@
 def test():
+    pass
"""
    code, out = run_diff_rules(diff)
    assert code == 0
    assert "diff_rules: OK" in out

def test_dev_null():
    diff = """--- /dev/null
+++ b/python/sglang/test.py
@@ -0,0 +1,2 @@
+def test():
+    pass
"""
    code, out = run_diff_rules(diff)
    assert code == 0
    assert "diff_rules: OK" in out

def test_fail_path_not_sglang():
    diff = """--- a/python/other/test.py
+++ b/python/other/test.py
@@ -1,1 +1,2 @@
 def test():
+    pass
"""
    code, out = run_diff_rules(diff)
    assert code == 1
    assert "FAIL Path" in out

def test_fail_path_model_loader():
    diff = """--- a/python/sglang/srt/model_loader/test.py
+++ b/python/sglang/srt/model_loader/test.py
@@ -1,1 +1,2 @@
 def test():
+    pass
"""
    code, out = run_diff_rules(diff)
    assert code == 1
    assert "FAIL Path" in out

def test_fail_path_suffix():
    diff = """--- a/python/sglang/test.txt
+++ b/python/sglang/test.txt
@@ -1,1 +1,2 @@
 def test():
+    pass
"""
    code, out = run_diff_rules(diff)
    assert code == 1
    assert "FAIL Path" in out

def test_fail_rename():
    diff = """rename from python/sglang/a.py
rename to python/sglang/b.py
"""
    code, out = run_diff_rules(diff)
    assert code == 1
    assert "FAIL Git: renames are forbidden" in out

def test_fail_mode():
    diff = """old mode 100644
new mode 100755
--- a/python/sglang/test.py
+++ b/python/sglang/test.py
"""
    code, out = run_diff_rules(diff)
    assert code == 1
    assert "FAIL Git: mode changes are forbidden" in out

def test_fail_binary():
    diff = """GIT binary patch
--- a/python/sglang/test.py
+++ b/python/sglang/test.py
"""
    code, out = run_diff_rules(diff)
    assert code == 1
    assert "FAIL Git: binary patches are forbidden" in out

def test_fail_line_length():
    diff = f"""--- a/python/sglang/test.py
+++ b/python/sglang/test.py
@@ -1,1 +1,2 @@
 def test():
+    {'a' * 401}
"""
    code, out = run_diff_rules(diff)
    assert code == 1
    assert "FAIL Line length" in out

def test_fail_forbidden_word():
    diff = """--- a/python/sglang/test.py
+++ b/python/sglang/test.py
@@ -1,1 +1,2 @@
 def test():
+    # kc-secret
"""
    code, out = run_diff_rules(diff)
    assert code == 1
    assert "FAIL Forbidden word" in out

def test_warnings():
    diff = """--- a/python/sglang/test.py
+++ b/python/sglang/test.py
@@ -1,1 +1,5 @@
 def test():
+    # DFlash 8 2048 rmsnorm
+    # cache speculative
-    # rmsnorm
"""
    code, out = run_diff_rules(diff)
    assert code == 0
    assert "WARN: potential change to DFlash" in out
    assert "WARN: potential change to expert count (8)" in out
    assert "WARN: potential change to sliding window (2048)" in out
    assert "WARN: potential norm removal detected" in out
    assert "WARN: potential global/cross-request cache" in out
