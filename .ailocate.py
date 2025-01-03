import os
import sys
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
import hashlib
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.console import Console
from pprint import pprint
import random

def dier(msg):
    pprint(msg)
    sys.exit(10)

console: Console = Console(
    force_interactive=True,
    soft_wrap=True,
    color_system="256",
    force_terminal=True
)

# Constants
DEFAULT_DB_PATH = os.path.expanduser('~/.ailocate_db')
DEFAULT_MODEL = "yolov5s.pt"
DEFAULT_THRESHOLD = 0.3
DEFAULT_DIR = "/"

supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

parser = argparse.ArgumentParser(description="YOLO File Indexer")
parser.add_argument("search", nargs="?", help="Search term for indexed results", default=None)
parser.add_argument("--index", action="store_true", help="Index images in the specified directory")
parser.add_argument("--dir", default=DEFAULT_DIR, help="Directory to search or index")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--delete_non_existing_files", action="store_true", help="Delete non-existing files")
parser.add_argument("--shuffle_index", action="store_true", help="Shuffle list of files before indexing")
parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use for detection")
parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold (0-1)")
parser.add_argument("--dbfile", default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
parser.add_argument("--stat", nargs="?", help="Display statistics for images or a specific file")
args = parser.parse_args()

console = Console()

def dbg(msg):
    if args.debug:
        console.log(f"[bold yellow]DEBUG:[/] {msg}")

try:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    if args.index:
        with console.status("[bold green]Loading torch...") as status:
            import torch
        with console.status("[bold green]Loading yolov5...") as status:
            import yolov5
except ModuleNotFoundError as e:
    console.print(f"[red]Module not found:[/] {e}")
    sys.exit(1)
except KeyboardInterrupt:
    console.print("\n[red]You pressed CTRL+C[/]")
    sys.exit(0)

def load_existing_images(conn):
    """Lädt alle Dateinamen und MD5-Hashes aus der Datenbank und gibt sie als Dictionary zurück."""
    cursor = conn.cursor()
    cursor.execute('''SELECT file_path, md5 FROM images''')
    rows = cursor.fetchall()
    # Gibt ein Dictionary zurück, das die Dateipfade mit den MD5-Hashes verknüpft
    return {row[0]: row[1] for row in rows}

def is_file_in_db(conn, file_path, existing_files):
    """Prüft, ob eine Datei bereits in der Datenbank existiert, unter Verwendung des vorab geladenen Dictionaries."""
    # Wenn der Dateipfad bereits im Dictionary vorhanden ist, ist die Datei bereits in der DB
    if file_path in existing_files:
        return True
    else:
        # Ansonsten führen wir die Datenbankabfrage durch
        cursor = conn.cursor()
        cursor.execute('''SELECT COUNT(*) FROM images WHERE file_path = ?''', (file_path,))
        return cursor.fetchone()[0] > 0

def get_md5(file_path):
    """Berechnet den MD5-Hash einer Datei."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def add_empty_image(conn, file_path):
    dbg(f"add_empty_image(conn, {file_path})")
    md5_hash = get_md5(file_path)

    # Überprüfen, ob die Datei bereits in der leeren Liste ist
    cursor = conn.cursor()
    cursor.execute('''SELECT md5 FROM empty_images WHERE file_path = ?''', (file_path,))
    existing_hash = cursor.fetchone()

    if existing_hash:
        # Wenn der Hash sich geändert hat, aktualisiere ihn
        if existing_hash[0] != md5_hash:
            cursor.execute('''UPDATE empty_images SET md5 = ? WHERE file_path = ?''', (md5_hash, file_path))
            conn.commit()
            dbg(f"Updated MD5 hash for {file_path}")
    else:
        # Wenn das Bild noch nicht in der Tabelle ist, füge es hinzu
        cursor.execute('''INSERT INTO empty_images (file_path, md5) VALUES (?, ?)''', (file_path, md5_hash))
        conn.commit()
        dbg(f"Added empty image: {file_path}")

# Datenbankstruktur für leere Bilder mit MD5-Hash
def init_database(db_path):
    dbg(f"init_database({db_path})")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY,
                        file_path TEXT UNIQUE,
                        size INTEGER,
                        created_at TEXT,
                        last_modified_at TEXT,
                        md5 TEXT
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY,
                        image_id INTEGER,
                        model TEXT,
                        label TEXT,
                        confidence REAL,
                        FOREIGN KEY(image_id) REFERENCES images(id)
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS empty_images (
                        file_path TEXT UNIQUE,
                        md5 TEXT
                    )''')  # Hinzugefügt: MD5-Hash für leere Bilder
    conn.commit()
    return conn

def calculate_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def add_image_metadata(conn, file_path):
    dbg(f"add_image_metadata(conn, {file_path})")
    cursor = conn.cursor()
    stats = os.stat(file_path)
    md5_hash = calculate_md5(file_path)
    created_at = datetime.fromtimestamp(stats.st_ctime).isoformat()
    last_modified_at = datetime.fromtimestamp(stats.st_mtime).isoformat()
    try:
        cursor.execute('''INSERT OR IGNORE INTO images (file_path, size, created_at, last_modified_at, md5)
                          VALUES (?, ?, ?, ?, ?)''', (file_path, stats.st_size, created_at, last_modified_at, md5_hash))
        conn.commit()
        return cursor.lastrowid, md5_hash
    except sqlite3.OperationalError as e:
        console.print(f"\n[red]Probably old table. Delete {args.dbfile}, error:[/] {e}")
        sys.exit(12)

def is_image_indexed(conn, file_path, model):
    dbg(f"is_image_indexed(conn, {file_path}, model)")
    cursor = conn.cursor()
    stats = os.stat(file_path)
    last_modified_at = datetime.fromtimestamp(stats.st_mtime).isoformat()
    cursor.execute('''SELECT COUNT(*) FROM images
                      JOIN detections ON images.id = detections.image_id
                      WHERE images.file_path = ?
                      AND detections.model = ?
                      AND images.last_modified_at = ?''',
                   (file_path, model, last_modified_at))
    return cursor.fetchone()[0] > 0

def add_detections(conn, image_id, model, detections):
    dbg(f"add_detections(conn, {image_id}, detections)")
    cursor = conn.cursor()
    for label, confidence in detections:
        cursor.execute('''INSERT INTO detections (image_id, model, label, confidence)
                          VALUES (?, ?, ?, ?)''', (image_id, model, label, confidence))
    conn.commit()

def find_images(directory, existing_files):
    dbg(f"find_images({directory}, existing_files)")
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_formats and file not in existing_files:
                yield os.path.join(root, file)

def analyze_image(model, image_path):
    dbg(f"analyze_image(model, {image_path})")
    try:
        console.print(f"[bright_yellow]Predicting {image_path}[/]")
        results = model(image_path)
        predictions = results.pred[0]
        detections = [(model.names[int(pred[5])], float(pred[4])) for pred in predictions if float(pred[4]) >= 0]
        return detections
    except (OSError, RuntimeError):
        return None

def process_image(image_path, model, conn, progress, progress_task):
    dbg(f"process_image({image_path}, model, conn)")

    image_id, md5_hash = add_image_metadata(conn, image_path)

    # Check for empty image (no detections found)
    detections = analyze_image(model, image_path)
    if detections:
        add_detections(conn, image_id, args.model, detections)
    else:
        # Mark image as "empty" and store MD5 hash in empty_images table
        add_empty_image(conn, image_path)

    progress.update(progress_task, advance=1)

def show_statistics(conn, file_path=None):
    if file_path:
        cursor = conn.cursor()
        cursor.execute('''SELECT detections.label, COUNT(*) FROM detections
                          JOIN images ON images.id = detections.image_id
                          WHERE images.file_path = ?
                          GROUP BY detections.label''', (file_path,))
        stats = cursor.fetchall()
        table = Table(title=f"Statistics for {file_path}")
        table.add_column("Label", justify="left", style="cyan")
        table.add_column("Count", justify="right", style="green")
        for row in stats:
            table.add_row(row[0], str(row[1]))
        console.print(table)
    else:
        cursor = conn.cursor()
        cursor.execute('''SELECT detections.label, COUNT(*) FROM detections
                          JOIN images ON images.id = detections.image_id
                          GROUP BY detections.label''')
        stats = cursor.fetchall()
        table = Table(title="Category Statistics")
        table.add_column("Label", justify="left", style="cyan")
        table.add_column("Count", justify="right", style="green")
        for row in stats:
            table.add_row(row[0], str(row[1]))
        console.print(table)

def delete_entries_by_filename(conn, file_path):
    """Löscht alle Einträge aus der Datenbank, die mit dem angegebenen Dateinamen verknüpft sind."""
    dbg(f"delete_entries_by_filename(conn, {file_path})")

    cursor = conn.cursor()

    # Lösche alle Detections für das Bild
    cursor.execute('''DELETE FROM detections WHERE image_id IN (SELECT id FROM images WHERE file_path = ?)''', (file_path,))

    # Lösche das Bild selbst aus der Tabelle 'images'
    cursor.execute('''DELETE FROM images WHERE file_path = ?''', (file_path,))

    # Lösche die MD5-Hash-Informationen aus der Tabelle 'empty_images' (falls vorhanden)
    cursor.execute('''DELETE FROM empty_images WHERE file_path = ?''', (file_path,))

    conn.commit()
    console.print(f"[red]Deleted all entries for {file_path}[/]")

def main():
    dbg(f"Arguments: {args}")

    conn = init_database(args.dbfile)

    existing_files = None

    if args.index or args.delete_non_existing_files:
        existing_files = load_existing_images(conn)

    if args.delete_non_existing_files:
        for file in existing_files:
            if not os.path.exists(file):
                delete_entries_by_filename(conn, file)
        existing_files = load_existing_images(conn)

    if args.index:
        model = yolov5.load(args.model)
        model.conf = 0

        image_paths = []

        with console.status("[bold green]Finding images...") as status:
            image_paths = list(find_images(args.dir, existing_files))
        total_images = len(image_paths)

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "[bold green]{task.completed}/{task.total} images",
            TimeElapsedColumn(),
            "[bold]Remaining[/]",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing images...", total=total_images)

            if args.shuffle_index:
                random.shuffle(image_paths)

            for image_path in image_paths:
                # Überprüfe, ob die Datei bereits in der Datenbank ist
                if is_file_in_db(conn, image_path, existing_files):
                    console.print(f"[green]Image {image_path} already in database. Skipping it.[/]")
                    progress.update(task, advance=1)
                else:
                    if is_image_indexed(conn, image_path, args.model):
                        console.print(f"[green]Image {image_path} already indexed. Skipping it.[/]")
                        progress.update(task, advance=1)
                    else:
                        process_image(image_path, model, conn, progress, task)
                        existing_files[image_path] = get_md5(image_path)

    if args.search:
        cursor = conn.cursor()
        cursor.execute('''SELECT images.file_path, detections.label, detections.confidence
                          FROM images JOIN detections ON images.id = detections.image_id
                          WHERE detections.label LIKE ?''', (f"%{args.search}%",))

        table = Table(title="Search Results")
        table.add_column("File Path", justify="left", style="cyan")
        table.add_column("Label", justify="center", style="magenta")
        table.add_column("Confidence", justify="right", style="green")

        results = cursor.fetchall()
        for row in results:
            conf = row[2]
            if conf >= args.threshold:
                table.add_row(*map(str, row))

        console.print(table)

    if args.stat:
        show_statistics(conn, args.stat if args.stat != "/" else None)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]You pressed CTRL+C[/]")
        sys.exit(0)
