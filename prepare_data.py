"""下载 UCI 原始文件，保存原始数据、校验信息和统一的 X/y 数组。"""
from pathlib import Path
import subprocess
import zipfile
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent / "data"
SOURCES = {
    "breast_cancer": ("breast-cancer-wisconsin/wdbc.data", 569, 30),
    "banknote": ("00267/data_banknote_authentication.txt", 1372, 4),
    "spambase": ("spambase/spambase.data", 4601, 57),
    "magic": ("magic/magic04.data", 19020, 10),
}
ARCHIVES = {
    "spambase": "https://archive.ics.uci.edu/static/public/94/spambase.zip",
    "magic": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
}


def main():
    ROOT.mkdir(exist_ok=True)
    manifest = {}
    for name, (remote, n, d) in SOURCES.items():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/" + remote
        path = ROOT / (name + ".csv")
        if not path.exists():
            # Windows curl 使用系统证书存储，保留 HTTPS 证书验证。
            temp = path.with_suffix(".download")
            download_url = ARCHIVES.get(name, url)
            subprocess.run(["curl.exe", "--fail", "--location", "--retry", "3",
                            "--max-time", "600", "--output", str(temp), download_url], check=True)
            if name in ARCHIVES:
                with zipfile.ZipFile(temp) as archive:
                    path.write_bytes(archive.read(remote.rsplit("/", 1)[-1]))
                temp.replace(ROOT / (name + ".zip"))
            else:
                temp.replace(path)
        rows = np.loadtxt(path, delimiter=",", dtype=str)
        if name == "breast_cancer":
            X, y = rows[:, 2:].astype(float), np.where(rows[:, 1] == "M", 1, -1)
        else:
            X = rows[:, :-1].astype(float)
            y = np.where(rows[:, -1] == ("g" if name == "magic" else "1"), 1, -1)
        assert X.shape == (n, d) and np.isfinite(X).all() and set(y) == {-1, 1}
        np.savez(ROOT / (name + ".npz"), X=X, y=y)
        manifest[name] = dict(url=url, download_url=ARCHIVES.get(name, url), shape=list(X.shape),
                              positive=int((y == 1).sum()), negative=int((y == -1).sum()),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(name, X.shape, flush=True)
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
