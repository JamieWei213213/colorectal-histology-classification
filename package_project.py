"""Package deliverables without caches, downloaded datasets or unrelated files."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
target = root.parent/'Colorectal_Histology_Classification_Project.zip'
files = [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
         and 'data' not in p.relative_to(root).parts and not p.name.endswith('.pyc')]
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, 'colorectal_histology/'+path.relative_to(root).as_posix())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert 'colorectal_histology/Colorectal_Histology_Classification.ipynb' in archive.namelist()
print(json.dumps({'archive':str(target), 'files':len(files), 'bytes':target.stat().st_size,
                  'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))
