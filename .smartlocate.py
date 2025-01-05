import uuid
import os
import sys
import argparse
import sqlite3
import random
from pprint import pprint
import time
from typing import Optional, Any, Generator

from pathlib import Path
from datetime import datetime
import hashlib
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.console import Console
import PIL
from PIL import Image
from sixel import converter

def dier(msg: Any) -> None:
    pprint(msg)
    sys.exit(10)

console: Console = Console(
    force_interactive=True,
    soft_wrap=True,
    color_system="256",
    force_terminal=True
)

MIN_CONFIDENCE_FOR_SAVING: float = 0.1
DEFAULT_DB_PATH: str = os.path.expanduser('~/.smartlocate_db')
DEFAULT_MODEL: str = "yolov5s.pt"
DEFAULT_THRESHOLD: float = 0.3
DEFAULT_DIR: str = "/"

supported_formats: set[str] = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

parser = argparse.ArgumentParser(description="YOLO File Indexer")
parser.add_argument("search", nargs="?", help="Search term for indexed results", default=None)
parser.add_argument("--index", action="store_true", help="Index images in the specified directory")
parser.add_argument("--size", type=int, default=400, help="Size to which the image should be resized (default: 400).")
parser.add_argument("--dir", default=DEFAULT_DIR, help="Directory to search or index")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--no_sixel", action="store_true", help="Hide sixel graphics")
parser.add_argument("--delete_non_existing_files", action="store_true", help="Delete non-existing files")
parser.add_argument("--shuffle_index", action="store_true", help="Shuffle list of files before indexing")
parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use for detection")
parser.add_argument("--describe", action="store_true", help="Enable image description")
parser.add_argument("--ocr", action="store_true", help="Enable OCR")
parser.add_argument("--ocr_lang", nargs='+', default=['de', 'en'], help="OCR languages, default: de, en. Accepts multiple languages.")
parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Confidence threshold (0-1)")
parser.add_argument("--max_ocr_size", type=int, default=5, help="Max-MB-Size for OCR in MB (default: 5)")
parser.add_argument("--dbfile", default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
parser.add_argument("--stat", nargs="?", help="Display statistics for images or a specific file")
args = parser.parse_args()

blip_model_name: str = "Salesforce/blip-image-captioning-large"
blip_processor: Any = None
blip_model: Any = None
reader: Any = None

console = Console()

def dbg(msg: Any) -> None:
    if args.debug:
        console.log(f"[bold yellow]DEBUG:[/] {msg}")

try:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    if args.index:
        with console.status("[bold green]Loading yolov5...") as load_status:
            import yolov5

        if args.ocr:
            with console.status("[bold green]Loading easyocr...") as load_status:
                import easyocr
            with console.status("[bold green]Loading reader...") as load_status:
                reader = easyocr.Reader(args.ocr_lang)
            with console.status("[bold green]Loading cv2...") as load_status:
                import cv2
        if args.describe:
            with console.status("[bold green]Loading transformers...") as load_status:
                import transformers

            with console.status("[bold green]Loading Blip-Transformers...") as load_status:
                from transformers import BlipProcessor, BlipForConditionalGeneration

            with console.status("[bold green]Loading Blip-Models...") as load_status:
                blip_processor = BlipProcessor.from_pretrained(blip_model_name)
                blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
except ModuleNotFoundError as e:
    console.print(f"[red]Module not found:[/] {e}")
    sys.exit(1)
except KeyboardInterrupt:
    console.print("\n[red]You pressed CTRL+C[/]")
    sys.exit(0)

def ocr_img(img: str) -> Optional[str]:
    try:
        if reader is None:
            console.print("[red]reader was not defined. Will not OCR.[/]")
            return None

        if os.path.exists(img):
            result = reader.readtext(img)

            return result

        console.print(f"[red]ocr_img: file {img} not found[/]")
        return None
    except (cv2.error, ValueError) as e:
        console.print(f"[red]ocr_img: file {img} caused an error: {e}[/]")
        return None

def resize_image(input_path: str, output_path: str, max_size: int) -> None:
    with Image.open(input_path) as img:
        img.thumbnail((max_size, max_size))
        img.save(output_path)

def display_sixel(image_path: str) -> None:
    unique_filename = f"/tmp/{uuid.uuid4().hex}_resized_image.png"

    try:
        resize_image(image_path, unique_filename, args.size)

        c = converter.SixelConverter(unique_filename)
        c.write(sys.stdout)
    except FileNotFoundError:
        console.print(f"[red]Could not find {image_path}[/]")
    finally:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)

def load_existing_images(conn: sqlite3.Connection) -> dict[Any, Any]:
    """Lädt alle Dateinamen und MD5-Hashes aus der Datenbank und gibt sie als Dictionary zurück."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT file_path, md5 FROM images
        UNION ALL
        SELECT file_path, md5 FROM ocr_results;
    ''')
    rows = cursor.fetchall()
    cursor.close()
    return {row[0]: row[1] for row in rows}

def is_file_in_img_desc_db(conn: sqlite3.Connection, file_path: str) -> bool:
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM image_description WHERE file_path = ?''', (file_path,))
    res = cursor.fetchone()[0]
    cursor.close()

    return res > 0

def is_file_in_ocr_db(conn: sqlite3.Connection, file_path: str) -> bool:
    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM ocr_results WHERE file_path = ?''', (file_path,))
    res = cursor.fetchone()[0]
    cursor.close()

    return res > 0

def is_file_in_yolo_db(conn: sqlite3.Connection, file_path: str, existing_files: Optional[dict]) -> bool:
    if existing_files and file_path in existing_files:
        return True

    cursor = conn.cursor()
    cursor.execute('''SELECT COUNT(*) FROM images WHERE file_path = ?''', (file_path,))
    res = cursor.fetchone()[0]
    cursor.close()

    return res > 0

def is_existing_detections_label(conn: sqlite3.Connection, label: str) -> bool:
    cursor = conn.cursor()
    cursor.execute('''SELECT label FROM detections WHERE label = ? LIMIT 1''', (label,))
    res = cursor.fetchone()  # Gibt entweder eine Zeile oder None zurück
    cursor.close()

    return res is not None  # Wenn eine Zeile zurückgegeben wurde, existiert das Label

def get_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def add_empty_image(conn: sqlite3.Connection, file_path: str) -> None:
    dbg(f"add_empty_image(conn, {file_path})")
    md5_hash = get_md5(file_path)

    cursor = conn.cursor()

    while True:
        try:
            cursor.execute('''SELECT md5 FROM empty_images WHERE file_path = ?''', (file_path,))
            existing_hash = cursor.fetchone()

            if existing_hash:
                if existing_hash[0] != md5_hash:
                    cursor.execute('''UPDATE empty_images SET md5 = ? WHERE file_path = ?''', (md5_hash, file_path))
                    conn.commit()
                    dbg(f"Updated MD5 hash for {file_path}")
            else:
                cursor.execute('''INSERT INTO empty_images (file_path, md5) VALUES (?, ?)''', (file_path, md5_hash))
                conn.commit()
                dbg(f"Added empty image: {file_path}")
            cursor.close()
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                console.print(f"\n[red]Error: {e}[/]")
                sys.exit(12)

def init_database(db_path: str) -> sqlite3.Connection:
    with console.status("[bold green]Initializing database...") as status:
        dbg(f"init_database({db_path})")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        status.update("[bold green]Creating table images...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                            id INTEGER PRIMARY KEY,
                            file_path TEXT UNIQUE,
                            size INTEGER,
                            created_at TEXT,
                            last_modified_at TEXT,
                            md5 TEXT
                        )''')
        status.update("[bold green]Created table images.")

        status.update("[bold green]Creating index images(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path)')
        status.update("[bold green]Created index images(file_path).")

        status.update("[bold green]Creating index images(md5)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_md5 ON images(md5)')
        status.update("[bold green]Created index images(md5).")

        status.update("[bold green]Creating table detections...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                            id INTEGER PRIMARY KEY,
                            image_id INTEGER,
                            model TEXT,
                            label TEXT,
                            confidence REAL,
                            FOREIGN KEY(image_id) REFERENCES images(id)
                        )''')
        status.update("[bold green]Created table detections.")

        status.update("[bold green]Creating index detections(image_id)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(image_id)')
        status.update("[bold green]Created index detections(image_id).")

        status.update("[bold green]Creating index detections(confidence)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(confidence)')
        status.update("[bold green]Created index detections(confidence).")

        status.update("[bold green]Creating index detections(image_id, model)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image_model ON detections(image_id, model)')
        status.update("[bold green]Created index detections(image_id, model).")

        status.update("[bold green]Creating index detections(label)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label);')
        status.update("[bold green]Created index detections(label).")

        status.update("[bold green]Creating table empty_images...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS empty_images (
                            file_path TEXT UNIQUE,
                            md5 TEXT
                        )''')
        status.update("[bold green]Created table empty_images.")

        status.update("[bold green]Creating index empty_images(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_empty_images_file_path ON empty_images(file_path)')
        status.update("[bold green]Created index empty_images(file_path).")

        status.update("[bold green]Creating table ocr_results...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS ocr_results (
                            id INTEGER PRIMARY KEY,
                            file_path TEXT UNIQUE,
                            extracted_text TEXT,
                            md5 TEXT
                        )''')
        status.update("[bold green]Created table ocr_results.")

        status.update("[bold green]Creating index ocr_results(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocr_results_file_path ON ocr_results(file_path)')
        status.update("[bold green]Created index ocr_results(file_path).")

        status.update("[bold green]Creating index ocr_results(md5)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocr_results_md5 ON ocr_results(md5)')
        status.update("[bold green]Created index ocr_results(md5).")

        status.update("[bold green]Creating table image_description...")
        cursor.execute('''CREATE TABLE IF NOT EXISTS image_description (
                            id INTEGER PRIMARY KEY,
                            file_path TEXT UNIQUE,
                            image_description TEXT,
                            md5 TEXT
                        )''')
        status.update("[bold green]Created table image_description.")

        status.update("[bold green]Creating index image_description(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_description_file_path ON image_description(file_path)')
        status.update("[bold green]Created index image_description(file_path).")

        status.update("[bold green]Creating index detections(label)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label)')
        status.update("[bold green]Created index detections(label).")

        status.update("[bold green]Creating index detections(image_id)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_image_id ON detections(image_id)')
        status.update("[bold green]Created index detections(image_id).")

        status.update("[bold green]Creating index images(file_path)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path)')
        status.update("[bold green]Created index images(file_path).")

        status.update("[bold green]Creating index detections(label, image_id)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_label_image_id ON detections(label, image_id)')
        status.update("[bold green]Created index detections(label, image_id).")

        status.update("[bold green]Creating index image_description_no_case...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_description_no_case ON image_description(image_description COLLATE NOCASE)')
        status.update("[bold green]Created index image_description_no_case.")

        status.update("[bold green]Creating index detections(label)...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label)')
        status.update("[bold green]Created index detections(label).")

        cursor.close()
        conn.commit()
        return conn

def calculate_md5(file_path: str) -> str:
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def execute_with_retry(conn: sqlite3.Connection, query: str, params: tuple) -> None:
    cursor = conn.cursor()

    while True:
        try:
            cursor.execute(query, params)
            break
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                raise e

    while True:
        try:
            cursor.close()
            conn.commit()
            break
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                raise e

def add_image_metadata(conn: sqlite3.Connection, file_path: str) -> tuple[int, str]:
    dbg(f"add_image_metadata(conn, {file_path})")
    cursor = conn.cursor()
    stats = os.stat(file_path)
    md5_hash = calculate_md5(file_path)
    created_at = datetime.fromtimestamp(stats.st_ctime).isoformat()
    last_modified_at = datetime.fromtimestamp(stats.st_mtime).isoformat()

    execute_with_retry(conn, '''INSERT OR IGNORE INTO images (file_path, size, created_at, last_modified_at, md5) VALUES (?, ?, ?, ?, ?)''', (file_path, stats.st_size, created_at, last_modified_at, md5_hash))

    cursor.execute('SELECT id FROM images WHERE file_path = ?', (file_path,))
    image_id = cursor.fetchone()[0]

    return image_id, md5_hash

def is_image_indexed(conn: sqlite3.Connection, file_path: str) -> bool:
    dbg(f"is_image_indexed(conn, {file_path})")

    while True:
        try:
            cursor = conn.cursor()
            stats = os.stat(file_path)
            last_modified_at = datetime.fromtimestamp(stats.st_mtime).isoformat()

            cursor.execute('''SELECT COUNT(*) FROM images
                               JOIN detections ON images.id = detections.image_id
                               WHERE images.file_path = ?
                               AND detections.model = ?
                               AND images.last_modified_at = ?''',
                           (file_path, args.model, last_modified_at))

            res = cursor.fetchone()[0]
            cursor.close()

            return res > 0
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                console.print(f"\n[red]Error: {e}[/]")
                sys.exit(12)
        except FileNotFoundError:
            return True

    return False

def add_detections(conn: sqlite3.Connection, image_id: int, model_name: str, detections: list) -> None:
    dbg(f"add_detections(conn, {image_id}, detections)")
    for label, confidence in detections:
        execute_with_retry(conn, '''INSERT INTO detections (image_id, model, label, confidence) VALUES (?, ?, ?, ?)''', (image_id, model_name, label, confidence))

def find_images(existing_files: dict) -> Generator:
    for root, _, files in os.walk(args.dir):
        for file in files:
            if Path(file).suffix.lower() in supported_formats and file not in existing_files:
                yield os.path.join(root, file)

def analyze_image(model: Any, image_path: str) -> Optional[list]:
    dbg(f"analyze_image(model, {image_path})")
    try:
        console.print(f"[bright_yellow]Predicting {image_path} with YOLO[/]")

        results = model(image_path)
        predictions = results.pred[0]
        detections = [(model.names[int(pred[5])], float(pred[4])) for pred in predictions if float(pred[4]) >= MIN_CONFIDENCE_FOR_SAVING]
        return detections
    except (OSError, RuntimeError):
        return None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/]")
        return None
    except PIL.Image.DecompressionBombError as e:
        console.print(f"[red]Error: {e}, probably the image is too large[/]")
        return None
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        return None

def process_image(image_path: str, model: Any, conn: sqlite3.Connection) -> None:
    dbg(f"process_image({image_path}, model, conn)")

    image_id, md5_hash = add_image_metadata(conn, image_path)

    detections = analyze_image(model, image_path)
    if detections:
        add_detections(conn, image_id, args.model, detections)
    else:
        add_empty_image(conn, image_path)

def show_statistics(conn: sqlite3.Connection, file_path: Optional[str]) -> None:
    if file_path:
        cursor = conn.cursor()
        cursor.execute('''SELECT detections.label, COUNT(*) FROM detections
                          JOIN images ON images.id = detections.image_id
                          WHERE images.file_path = ?
                          AND detections.confidence >= ?
                          GROUP BY detections.label''', (file_path, args.threshold,))
        stats = cursor.fetchall()
        cursor.close()
        table = Table(title=f"Statistics for {file_path} with confidence {args.threshold}")
        table.add_column("Label", justify="left", style="cyan")
        table.add_column("Count", justify="right", style="green")
        for row in stats:
            table.add_row(row[0], str(row[1]))
        console.print(table)
    else:
        cursor = conn.cursor()
        cursor.execute('''SELECT detections.label, COUNT(*) FROM detections
                          JOIN images ON images.id = detections.image_id
                          WHERE detections.confidence >= ?
                          GROUP BY detections.label''', (args.threshold,))
        stats = cursor.fetchall()
        cursor.close()
        table = Table(title="Category Statistics")
        table.add_column("Label", justify="left", style="cyan")
        table.add_column("Count", justify="right", style="green")
        for row in stats:
            table.add_row(row[0], str(row[1]))
        console.print(table)

def delete_entries_by_filename(conn: sqlite3.Connection, file_path: str) -> None:
    """Löscht alle Einträge aus der Datenbank, die mit dem angegebenen Dateinamen verknüpft sind."""
    dbg(f"delete_entries_by_filename(conn, {file_path})")

    while True:
        try:
            cursor = conn.cursor()

            with console.status("[bold green]Deleting files from DB that do not exist...") as delete_status:
                delete_status.update(f"[bold green]Deleting detections for {file_path}...")
                cursor.execute('''DELETE FROM detections WHERE image_id IN (SELECT id FROM images WHERE file_path = ?)''', (file_path,))
                delete_status.update(f"[bold green]Deleted from detections for {file_path}.")

                delete_status.update(f"[bold green]Deleting from images for {file_path}...")
                cursor.execute('''DELETE FROM images WHERE file_path = ?''', (file_path,))
                delete_status.update(f"[bold green]Deleted from images for {file_path}.")

                delete_status.update(f"[bold green]Deleting from empty_images for {file_path}...")
                cursor.execute('''DELETE FROM empty_images WHERE file_path = ?''', (file_path,))
                delete_status.update(f"[bold green]Deleted from empty_images for {file_path}.")

                delete_status.update(f"[bold green]Deleting from ocr_results for {file_path}...")
                cursor.execute('''DELETE FROM ocr_results WHERE file_path = ?''', (file_path,))
                delete_status.update(f"[bold green]Deleted from ocr_results for {file_path}.")

                delete_status.update(f"[bold green]Deleting from image_description for {file_path}...")
                cursor.execute('''DELETE FROM image_description WHERE file_path = ?''', (file_path,))
                delete_status.update(f"[bold green]Deleted from image_description for {file_path}.")

                cursor.close()
                conn.commit()

                console.print(f"[red]Deleted all entries for {file_path}[/]")
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                console.print("[yellow]Database is locked, retrying...[/]")
                time.sleep(1)
            else:
                console.print(f"\n[red]Error: {e}[/]")
                sys.exit(12)

def delete_non_existing_files(conn: sqlite3.Connection, existing_files: Optional[dict]) -> Optional[dict]:
    if existing_files:
        for file in existing_files:
            if not os.path.exists(file):
                delete_entries_by_filename(conn, file)
        existing_files = load_existing_images(conn)

    return existing_files

def add_description(conn: sqlite3.Connection, file_path: str, desc: str) -> None:
    dbg(f"add_description(conn, {file_path}, <desc>)")
    md5_hash = get_md5(file_path)
    execute_with_retry(conn, '''INSERT INTO image_description (file_path, image_description, md5) VALUES (?, ?, ?)''', (file_path, desc, md5_hash))

def add_ocr_result(conn: sqlite3.Connection, file_path: str, extracted_text: str) -> None:
    dbg(f"add_ocr_result(conn, {file_path}, <extracted_text>)")
    md5_hash = get_md5(file_path)
    execute_with_retry(conn, '''INSERT INTO ocr_results (file_path, extracted_text, md5) VALUES (?, ?, ?)''', (file_path, extracted_text, md5_hash))

def search_yolo(conn: sqlite3.Connection) -> None:
    yolo_results = None

    if not is_existing_detections_label(conn, args.search):
        return

    with console.status("[bold green]Searching through YOLO-results...") as status:
        cursor = conn.cursor()
        cursor.execute('''SELECT images.file_path, detections.label, detections.confidence
                          FROM images JOIN detections ON images.id = detections.image_id
                          WHERE detections.label LIKE ? GROUP BY images.file_path''', (f"%{args.search}%",))
        yolo_results = cursor.fetchall()
        cursor.close()

    if not args.no_sixel:
        for row in yolo_results:
            conf = row[2]
            if conf >= args.threshold:
                print(f"{row[0]} (certainty: {conf:.2f})")
                display_sixel(row[0])
                print("\n")
    else:
        table = Table(title="Search Results")
        table.add_column("File Path", justify="left", style="cyan")
        table.add_column("Label", justify="center", style="magenta")
        table.add_column("Confidence", justify="right", style="green")
        for row in yolo_results:
            conf = row[2]
            if conf >= args.threshold:
                table.add_row(*map(str, row))

        if len(yolo_results):
            console.print(table)

def search_description(conn: sqlite3.Connection) -> None:
    ocr_results = None

    with console.status("[bold green]Searching through descriptions...") as status:
        cursor = conn.cursor()
        cursor.execute('''SELECT file_path, image_description
                          FROM image_description
                          WHERE image_description LIKE ? COLLATE NOCASE''', (f"%{args.search}%",))
        ocr_results = cursor.fetchall()
        cursor.close()

    if not args.no_sixel:
        for row in ocr_results:
            print(f"File: {row[0]}\nDescription:\n{row[1]}\n")
            display_sixel(row[0])
            print("\n")
    else:
        table = Table(title="OCR Search Results")
        table.add_column("File Path", justify="left", style="cyan")
        table.add_column("Extracted Text", justify="center", style="magenta")
        for row in ocr_results:
            file_path, extracted_text = row
            table.add_row(file_path, extracted_text)

        if len(ocr_results):
            console.print(table)

def search_ocr(conn: sqlite3.Connection) -> None:
    ocr_results = None

    with console.status("[bold green]Searching through OCR results...") as status:
        cursor = conn.cursor()
        cursor.execute('''SELECT file_path, extracted_text
                          FROM ocr_results
                          WHERE extracted_text LIKE ? COLLATE NOCASE''', (f"%{args.search}%",))
        ocr_results = cursor.fetchall()
        cursor.close()

    if not args.no_sixel:
        for row in ocr_results:
            print(f"File: {row[0]}\nExtracted Text:\n{row[1]}\n")
            display_sixel(row[0])
            print("\n")
    else:
        table = Table(title="OCR Search Results")
        table.add_column("File Path", justify="left", style="cyan")
        table.add_column("Extracted Text", justify="center", style="magenta")
        for row in ocr_results:
            file_path, extracted_text = row
            table.add_row(file_path, extracted_text)

        if len(ocr_results):
            console.print(table)

def search(conn: sqlite3.Connection) -> None:
    search_yolo(conn)

    search_ocr(conn)

    search_description(conn)

def yolo_file(conn: sqlite3.Connection, image_path: str, existing_files: Optional[dict], model: Any) -> None:
    if model is None:
        return

    if is_file_in_yolo_db(conn, image_path, existing_files):
        console.print(f"[green]Image {image_path} already in yolo-database. Skipping it.[/]")
    else:
        if is_image_indexed(conn, image_path):
            console.print(f"[green]Image {image_path} already indexed. Skipping it.[/]")
        else:
            process_image(image_path, model, conn)
            if existing_files is not None:
                existing_files[image_path] = get_md5(image_path)

def get_image_description(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")

        if blip_processor is None:
            console.print("blip_processor was none. Cannot describe image.")
            return ""

        if blip_model is None:
            console.print("blip_model was none. Cannot describe image.")
            return ""

        inputs = blip_processor(images=image, return_tensors="pt")

        outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)

        return caption
    except PIL.Image.DecompressionBombError as e:
        console.print(f"File {image_path} failed with error {e}")
        return ""


def describe_img(conn: sqlite3.Connection, image_path: str) -> None:
    if args.describe:
        if is_file_in_img_desc_db(conn, image_path):
            console.print(f"[green]Image {image_path} already in image-description-database. Skipping it.[/]")
        else:
            try:
                image_description = get_image_description(image_path)
                if image_description:
                    console.print(f"[green]Saved description '{image_description}' for {image_path}[/]")
                    add_description(conn, image_path, image_description)
                else:
                    console.print(f"[yellow]Image {image_path} could not be described. Saving it as empty.[/]")
                    add_description(conn, image_path, "")

            except FileNotFoundError:
                console.print(f"[red]File {image_path} not found[/]")

def ocr_file(conn: sqlite3.Connection, image_path: str) -> None:
    if args.ocr:
        if is_file_in_ocr_db(conn, image_path):
            console.print(f"[green]Image {image_path} already in ocr-database. Skipping it.[/]")
        else:
            try:
                file_size = os.path.getsize(image_path)

                if file_size < args.max_ocr_size * 1024 * 1024:
                    extracted_text = ocr_img(image_path)
                    if extracted_text:
                        texts = [item[1] for item in extracted_text]
                        text = " ".join(texts)
                        if text:
                            add_ocr_result(conn, image_path, text)
                            console.print(f"[green]Saved OCR for {image_path}[/]")
                        else:
                            console.print(f"[yellow]Image {image_path} contains no text. Saving it as empty.[/]")
                            add_ocr_result(conn, image_path, "")
                    else:
                        console.print(f"[yellow]Image {image_path} contains no text. Saving it as empty.[/]")
                        add_ocr_result(conn, image_path, "")

                else:
                    console.print(f"[red]Image {image_path} is too large. Will skip OCR. Max-Size: {args.max_ocr_size}MB, is {file_size / 1024 / 1024}MB[/]")
            except FileNotFoundError:
                console.print(f"[red]File {image_path} not found[/]")

def main() -> None:
    dbg(f"Arguments: {args}")

    conn = init_database(args.dbfile)

    existing_files = None

    if args.index or args.delete_non_existing_files:
        existing_files = load_existing_images(conn)

    if args.delete_non_existing_files:
        existing_files = delete_non_existing_files(conn, existing_files)

    if args.index:
        model = yolov5.load(args.model)
        model.conf = 0

        image_paths = []

        with console.status(f"[bold green]Finding images in {args.dir}...") as status:
            if existing_files is not None:
                image_paths = list(find_images(existing_files))
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
                if os.path.exists(image_path):
                    yolo_file(conn, image_path, existing_files, model)
                    ocr_file(conn, image_path)
                    describe_img(conn, image_path)
                else:
                    console.print(f"[red]Could not find {image_path}[/]")

                progress.update(task, advance=1)

    if args.search:
        search(conn)

    if args.stat:
        show_statistics(conn, args.stat if args.stat != "/" else None)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]You pressed CTRL+C[/]")
        sys.exit(0)