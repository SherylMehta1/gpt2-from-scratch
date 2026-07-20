from urllib.request import urlretrieve
from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

urlretrieve(url, data_dir / "input.txt")

print("Dataset downloaded!")