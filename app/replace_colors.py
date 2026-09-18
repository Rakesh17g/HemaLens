import glob
import os
import re

pages_dir = r"c:\Users\raki_\OneDrive\Desktop\AIML\app"
files = glob.glob(os.path.join(pages_dir, "**", "*.py"), recursive=True)

css_replacements = [
    (r"color:var(--text)\b", r"color:var(--text)"),
    (r"color:var(--subtext)\b", r"color:var(--subtext)"),
    (r"color:var(--border)\b", r"color:var(--border)"),
]

plotly_replacements = [
    (r'"#F5F5F5"', r'"#F5F5F5"'),
    (r'"#888888"', r'"#888888"'),
    (r'"#050505"', r'"#050505"'),
    (r'"#111111"', r'"#111111"'),
    (r'"#2A2A2A"', r'"#2A2A2A"'),
    (r'"#a5f9ef"', r'"#a5f9ef"'),
    (r'"rgba\(99,102,241,0.18\)"', r'"rgba(165,249,239,0.18)"'),
]

for f in files:
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()

    original = content
    for old, new in css_replacements:
        content = re.sub(old, new, content)

    for old, new in plotly_replacements:
        content = re.sub(old, new, content)

    if original != content:
        with open(f, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Updated {f}")
