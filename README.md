# 📚 Bibliography Status Checker

A Python tool to check your reading status and library availability for an author’s works.  
Supports StoryGraph exports, Libby/OverDrive library lookups, and Open Library for bibliographic data.  

---

## ✨ Features
- Pulls **author’s full bibliography** from Open Library.  
- **Deduplicates** messy editions (“hardcover”, “sneak peek”, etc.).  
- Filters out **non-English editions** (configurable).  
- Cross-checks with your **StoryGraph export** to mark as:
  - `Read`
  - `Currently Reading`
  - `To-Read`
  - `Unread`  
- Queries **library availability** (Libby/OverDrive) when enabled.  
- Multiple output modes:
  - `--status-only` → only shows Title + Status (fast).  
  - `--only unread` / `--only to-read` → filters before checking availability (faster).  
  - `--strict-english` → require Open Library confirmation for English (slower, cleanest).  

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/bibliography-status.git
cd bibliography-status
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🔧 Usage

### Basic
```bash
python bibliography_status.py "T. Kingfisher"
```

### Status only (no availability lookups)
```bash
python bibliography_status.py "T. Kingfisher" --status-only
```

### Filter by status
```bash
python bibliography_status.py "T. Kingfisher" --only unread
```

### Stricter language filtering
```bash
python bibliography_status.py "T. Kingfisher" --strict-english
```

---

## 📂 Demo Files

Demo files are included so you can test the script without exposing your real data:

- `Demo StoryGraph.csv` – sample StoryGraph export with a few books.  
- `Demo libraries.txt` – example libraries config for Libby/OverDrive.  
- `Demo authors.txt` – list of authors to process.  

---

## 📝 Example Output

```
================================================================================
Author: T. Kingfisher
================================================================================
Title                                                                                                                   Status           
----------------------------------------------------------------------------------------------------------------------  -----------------
A House With Good Bones                                                                                                 Read
Minor Mage                                                                                                              Unread
Nettle & Bone                                                                                                           Currently Reading
Swordheart                                                                                                              Read
The Hollow Places                                                                                                       To-Read
```

---

## ⚖️ License
MIT License – see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing
Pull requests welcome! For major changes, please open an issue first to discuss what you’d like to change.  
