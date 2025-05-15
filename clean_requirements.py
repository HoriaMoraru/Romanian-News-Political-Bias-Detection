from pathlib import Path

raw_file = Path("raw_requirements.txt")
output_file = Path("requirements.txt")

excluded = {"pywin32", "pypiwin32", "pyreadline3", "colorama"}

with raw_file.open("r", encoding="utf-8-sig") as f, output_file.open("w", encoding="utf-8", newline="\n") as out:
    for line in f:
        if any(pkg.lower() in line.lower() for pkg in excluded):
            continue
        if "@" in line:
            out.write(line)
        elif "==" in line:
            name, version = line.strip().split("==")
            out.write(f"{name}>={version}\n")
        else:
            out.write(line)
