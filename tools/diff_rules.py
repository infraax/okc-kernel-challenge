import sys
import os
import re

MAX_SIZE_BYTES = 737280
MAX_LINE_LENGTH = 400
ALLOWED_SUFFIXES = {'.py', '.cu', '.cuh', '.h', '.hpp', '.c', '.cc', '.cpp', '.json'}
FORBIDDEN_WORDS = [
    'kc-secret', 'KC_RESULT', 'KC_SERVING', 'KC_GATE',
    '/kc-job', 'result_secret', 'serviceaccount', 'SIMULATE'
]

def check_diff(filepath):
    try:
        size = os.path.getsize(filepath)
    except OSError as e:
        print(f"FAIL OS: could not read file {filepath}: {e}")
        sys.exit(1)

    if size > MAX_SIZE_BYTES:
        print(f"FAIL Size: patch is {size} bytes, limit is {MAX_SIZE_BYTES} bytes")
        sys.exit(1)

    with open(filepath, 'rb') as f:
        # read line by line to handle potential decoding issues in binary files, though it shouldn't be binary
        try:
            content = f.read().decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            print("FAIL Encoding: patch is not valid UTF-8")
            sys.exit(1)

    lines = content.split('\n')
    files_touched = set()

    # Path checking
    for line in lines:
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            path = line[6:].strip()
            if path == '/dev/null':
                continue

            files_touched.add(path)

            if not path.startswith('python/sglang/'):
                print(f"FAIL Path: modifies {path}, only paths under python/sglang/ are allowed")
                sys.exit(1)

            if path.startswith('python/sglang/srt/model_loader/'):
                print(f"FAIL Path: modifies {path}, changes under python/sglang/srt/model_loader/ are forbidden")
                sys.exit(1)

            suffix = os.path.splitext(path)[1]
            if suffix not in ALLOWED_SUFFIXES:
                print(f"FAIL Path: {path} has forbidden suffix, only {', '.join(ALLOWED_SUFFIXES)} are allowed")
                sys.exit(1)

        # Prohibited git patch features
        if line.startswith('rename from') or line.startswith('rename to'):
            print(f"FAIL Git: renames are forbidden: {line}")
            sys.exit(1)
        if line.startswith('old mode') or line.startswith('new mode'):
            print(f"FAIL Git: mode changes are forbidden: {line}")
            sys.exit(1)
        if line.startswith('GIT binary patch') or line.startswith('Binary files'):
            print(f"FAIL Git: binary patches are forbidden")
            sys.exit(1)

        # Added lines constraints
        if line.startswith('+') and not line.startswith('+++ '):
            added_text = line[1:]
            if len(added_text) > MAX_LINE_LENGTH:
                print(f"FAIL Line length: added line > {MAX_LINE_LENGTH} characters")
                sys.exit(1)

            for word in FORBIDDEN_WORDS:
                if word in added_text:
                    print(f"FAIL Forbidden word: found '{word}' in added lines")
                    sys.exit(1)

        # Warnings
        if line.startswith('+') or line.startswith('-'):
            if 'DFlash' in line:
                print("WARN: potential change to DFlash weight loading detected")
            if re.search(r'\b8\b', line):
                print("WARN: potential change to expert count (8) detected")
            if '2048' in line:
                print("WARN: potential change to sliding window (2048) detected")
            if line.startswith('-') and re.search(r'rmsnorm', line, re.IGNORECASE):
                print("WARN: potential norm removal detected")
            if re.search(r'cache', line, re.IGNORECASE) and (re.search(r'speculative', line, re.IGNORECASE) or re.search(r'ngram', line, re.IGNORECASE)):
                print("WARN: potential global/cross-request cache added in speculative/ngram code detected")

    print(f"diff_rules: OK ({size} bytes, {len(files_touched)} files)")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python diff_rules.py <patch.diff>")
        sys.exit(1)

    check_diff(sys.argv[1])
