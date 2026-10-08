"""
Download and parse the real Apache SpamAssassin public mail corpus.

The script downloads the 2003 public-corpus archives:
    easy_ham, easy_ham_2, hard_ham, spam, spam_2

It extracts the raw email messages and converts them to a CSV containing:
    label, subject, text, source_group

No synthetic email is generated.
"""

from pathlib import Path
from urllib.request import urlopen, Request
import tarfile
import hashlib
import csv
import re
from email import policy
from email.parser import BytesParser
from html import unescape

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
RAW_DIR = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")
CSV_PATH = PROCESSED_DIR / "emails.csv"

ARCHIVES = {
    "easy_ham": "20030228_easy_ham.tar.bz2",
    "easy_ham_2": "20030228_easy_ham_2.tar.bz2",
    "hard_ham": "20030228_hard_ham.tar.bz2",
    "spam": "20030228_spam.tar.bz2",
    "spam_2": "20030228_spam_2.tar.bz2",
}


def download_file(url: str, destination: Path):
    if destination.exists():
        print(f"[SKIP] {destination.name} already exists")
        return

    print(f"[DOWNLOAD] {url}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response, open(destination, "wb") as f:
        total = response.headers.get("Content-Length")
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / int(total) * 100
                print(f"\r        {pct:5.1f}%", end="")
    print()


def extract_archive(archive_path: Path, destination: Path):
    marker = destination / f".extracted_{archive_path.stem}"
    if marker.exists():
        print(f"[SKIP] {archive_path.name} already extracted")
        return

    print(f"[EXTRACT] {archive_path.name}")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(destination)
    marker.touch()


def html_to_text(value: str) -> str:
    """Simple HTML-to-text conversion without external dependencies."""
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    value = unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def decode_part(part):
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = str(part.get_payload()).encode("utf-8", errors="ignore")
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    except Exception:
        try:
            return str(part.get_payload())
        except Exception:
            return ""


def extract_email_text(raw_bytes: bytes):
    """Return subject and useful textual body from a raw RFC email."""
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    except Exception:
        return "", ""

    subject = str(msg.get("subject", ""))

    plain_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue

            disposition = str(part.get_content_disposition() or "").lower()
            if disposition == "attachment":
                continue

            content_type = part.get_content_type()
            if content_type == "text/plain":
                plain_parts.append(decode_part(part))
            elif content_type == "text/html":
                html_parts.append(decode_part(part))
    else:
        content_type = msg.get_content_type()
        content = decode_part(msg)
        if content_type == "text/html":
            html_parts.append(content)
        else:
            plain_parts.append(content)

    if plain_parts:
        body = "\n".join(plain_parts)
    else:
        body = "\n".join(html_to_text(x) for x in html_parts)

    # Keep the subject because it is highly informative for email classification.
    text = f"Subject: {subject}\n{body}"
    text = re.sub(r"\s+", " ", text).strip()

    return subject, text


def iter_messages(extracted_root: Path):
    """Yield (group, path) for files inside extracted corpus directories."""
    for group in ARCHIVES:
        candidates = list(extracted_root.rglob(group))
        if not candidates:
            # tarballs use easy_ham, easy_ham_2, etc. as top-level dirs
            candidates = list(extracted_root.rglob(group.replace("_", "_")))
        if not candidates:
            continue

        group_dir = next((p for p in candidates if p.is_dir()), None)
        if group_dir:
            for path in group_dir.rglob("*"):
                if path.is_file():
                    yield group, path


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    for filename in ARCHIVES.values():
        download_file(BASE_URL + filename, RAW_DIR / filename)

    # Extract
    for filename in ARCHIVES.values():
        extract_archive(RAW_DIR / filename, RAW_DIR)

    # Parse
    rows = []
    seen_hashes = set()
    counts = {"spam": 0, "ham": 0, "skipped": 0, "duplicates": 0}

    for group, path in iter_messages(RAW_DIR):
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen_hashes:
                counts["duplicates"] += 1
                continue
            seen_hashes.add(digest)

            subject, text = extract_email_text(raw)
            if len(text.strip()) < 5:
                counts["skipped"] += 1
                continue

            label = "spam" if group.startswith("spam") else "ham"
            rows.append({
                "label": label,
                "subject": subject,
                "text": text,
                "source_group": group,
            })
            counts[label] += 1
        except Exception as exc:
            counts["skipped"] += 1
            print(f"[WARN] Could not parse {path}: {exc}")

    if not rows:
        raise RuntimeError("No emails were parsed. Check the downloaded archives.")

    # Stable ordering makes runs reproducible.
    rows.sort(key=lambda x: (x["label"], x["source_group"], x["text"]))

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "subject", "text", "source_group"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDataset created:", CSV_PATH)
    print("Messages:", len(rows))
    print("Ham:", counts["ham"])
    print("Spam:", counts["spam"])
    print("Skipped:", counts["skipped"])
    print("Duplicates:", counts["duplicates"])


if __name__ == "__main__":
    main()
