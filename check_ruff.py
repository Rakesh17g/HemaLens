import json, subprocess
res = subprocess.run(['python', '-m', 'ruff', 'check', 'src/', 'app/', 'tests/', '--output-format', 'json'], capture_output=True, text=True)
try:
    data = json.loads(res.stdout)
    for e in data:
        print(f"{e['filename']}:{e['location']['row']} {e['code']} {e['message']}")
except Exception as e:
    print("Error parsing json", e)
    print(res.stdout)
