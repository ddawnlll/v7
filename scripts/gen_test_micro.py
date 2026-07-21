"""Generate synthetic OKX-format test data for pipeline validation."""
import csv, zipfile, os
from pathlib import Path

sample_dir = Path('data/raw/okx/sample/BTC-USDT-SWAP')
sample_dir.mkdir(parents=True, exist_ok=True)
(sample_dir / 'trades').mkdir(exist_ok=True)
(sample_dir / 'book').mkdir(exist_ok=True)

base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC

# --- Trades: 1 hour, 4 trades/min = 240 rows, headerless CSV ---
trades = []
trade_id = 100000
for minute in range(60):
    for sec in range(0, 60, 15):
        ts = base_ts + minute * 60000 + sec * 1000
        px = 42000.0 + (minute % 20) * 10
        sz = 0.01 * ((sec // 15) + 1)
        side = 'buy' if (minute + sec) % 2 == 0 else 'sell'
        trades.append([str(ts), str(trade_id), str(px), str(sz), side])
        trade_id += 1

csv_path = sample_dir / 'trades' / '2024-01-01.csv'
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerows(trades)
zip_path = sample_dir / 'trades' / '2024-01-01.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(str(csv_path), 'BTC-USDT-SWAP-trades-2024-01-01.csv')
os.remove(csv_path)
print(f'Trades: {len(trades)} rows -> {zip_path}')

# --- Books: L5 snapshot every second, 3600 rows, headerless CSV ---
# Format: ts, seq_id, checksum, bid_px1, bid_sz1, ..., bid_px5, bid_sz5, ask_px1, ask_sz1, ..., ask_px5, ask_sz5
book_rows = []
for seq in range(3600):
    ts = base_ts + seq * 1000
    mid = 42000.0 + ((seq // 60) % 20) * 10
    row = [str(ts), str(seq), '0']
    for i in range(5):
        row.append(f'{mid - (i+1)*5:.1f}')  # bid px
        row.append('1.0')                     # bid sz
    for i in range(5):
        row.append(f'{mid + (i+1)*5:.1f}')  # ask px
        row.append('1.0')                     # ask sz
    book_rows.append(row)

book_csv = sample_dir / 'book' / '2024-01-01.csv'
with open(book_csv, 'w', newline='') as f:
    csv.writer(f).writerows(book_rows)
book_zip = sample_dir / 'book' / '2024-01-01.zip'
with zipfile.ZipFile(book_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(str(book_csv), 'BTC-USDT-SWAP-book-2024-01-01.csv')
os.remove(book_csv)
print(f'Books:  {len(book_rows)} rows -> {book_zip}')
print('Done.')
