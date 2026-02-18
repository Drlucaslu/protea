#!/usr/bin/env python3
"""Ring 2 — Generation 407: Intelligent File Curator & Batch Optimizer

Focus: Proactive file discovery, smart categorization, delivery optimization, and pattern learning.
Strategy: Monitor filesystem, learn file patterns, optimize batch composition, predict delivery timing.
"""

import os
import pathlib
import time
import json
import hashlib
import mimetypes
from threading import Thread, Event
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re

HEARTBEAT_INTERVAL = 2


def heartbeat_loop(heartbeat_path: pathlib.Path, pid: int, stop_event: Event) -> None:
    """Dedicated heartbeat thread - CRITICAL for survival."""
    while not stop_event.is_set():
        try:
            heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
        except Exception:
            pass
        time.sleep(HEARTBEAT_INTERVAL)


class FileSignature:
    """Compute and analyze file signatures."""
    
    @staticmethod
    def compute_hash(filepath: pathlib.Path, quick: bool = True) -> str:
        """Compute file hash (quick mode uses first/last 8KB + size)."""
        try:
            stat = filepath.stat()
            if quick and stat.st_size > 16384:
                hash_obj = hashlib.sha256()
                with open(filepath, 'rb') as f:
                    hash_obj.update(f.read(8192))
                    f.seek(-8192, 2)
                    hash_obj.update(f.read(8192))
                    hash_obj.update(str(stat.st_size).encode())
                return hash_obj.hexdigest()[:16]
            else:
                hash_obj = hashlib.sha256()
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        hash_obj.update(chunk)
                return hash_obj.hexdigest()[:16]
        except Exception:
            return "error"
    
    @staticmethod
    def extract_metadata(filepath: pathlib.Path) -> dict:
        """Extract comprehensive file metadata."""
        try:
            stat = filepath.stat()
            mime_type, _ = mimetypes.guess_type(str(filepath))
            
            # Extract content hints from filename
            name_lower = filepath.stem.lower()
            content_hints = {
                "has_date": bool(re.search(r'\d{4}[-_]\d{2}[-_]\d{2}', name_lower)),
                "has_version": bool(re.search(r'v\d+\.\d+|\d+\.\d+\.\d+', name_lower)),
                "is_draft": any(word in name_lower for word in ['draft', 'temp', 'tmp', 'wip']),
                "is_final": any(word in name_lower for word in ['final', 'release', 'prod']),
                "has_number": bool(re.search(r'\d+', name_lower))
            }
            
            return {
                "path": filepath,
                "name": filepath.name,
                "stem": filepath.stem,
                "ext": filepath.suffix.lower(),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "ctime": stat.st_ctime,
                "mime_type": mime_type or "unknown",
                "content_hints": content_hints,
                "relative_path": None  # Set later
            }
        except Exception as e:
            return {"error": str(e)}


class FileCategorizationEngine:
    """Intelligent file categorization with learning."""
    
    CATEGORY_RULES = {
        "documents": {
            "extensions": [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".odt", ".ods"],
            "keywords": ["report", "document", "presentation", "sheet", "slide"],
            "priority": 10
        },
        "images": {
            "extensions": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".tiff"],
            "keywords": ["photo", "image", "pic", "screenshot"],
            "priority": 8
        },
        "code": {
            "extensions": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".go", ".rs", ".rb"],
            "keywords": ["script", "source", "code", "program"],
            "priority": 7
        },
        "data": {
            "extensions": [".json", ".csv", ".xml", ".yaml", ".yml", ".sql", ".db"],
            "keywords": ["data", "database", "config", "settings"],
            "priority": 6
        },
        "archives": {
            "extensions": [".zip", ".tar", ".gz", ".rar", ".7z", ".bz2"],
            "keywords": ["archive", "backup", "package"],
            "priority": 5
        },
        "media": {
            "extensions": [".mp4", ".avi", ".mov", ".mp3", ".wav", ".flac"],
            "keywords": ["video", "audio", "music", "sound"],
            "priority": 4
        },
        "text": {
            "extensions": [".txt", ".md", ".rst", ".log"],
            "keywords": ["note", "readme", "log", "text"],
            "priority": 3
        }
    }
    
    def __init__(self):
        self.category_stats = defaultdict(lambda: {"count": 0, "total_size": 0})
        self.pattern_frequency = Counter()
    
    def categorize(self, file_meta: dict) -> str:
        """Categorize file based on rules and learned patterns."""
        ext = file_meta.get("ext", "")
        name_lower = file_meta.get("stem", "").lower()
        
        # Rule-based categorization
        for category, rules in self.CATEGORY_RULES.items():
            if ext in rules["extensions"]:
                self._update_stats(category, file_meta)
                return category
            
            if any(keyword in name_lower for keyword in rules["keywords"]):
                self._update_stats(category, file_meta)
                return category
        
        self._update_stats("other", file_meta)
        return "other"
    
    def _update_stats(self, category: str, file_meta: dict):
        """Update category statistics."""
        self.category_stats[category]["count"] += 1
        self.category_stats[category]["total_size"] += file_meta.get("size", 0)
    
    def get_priority_score(self, category: str, file_meta: dict) -> float:
        """Calculate delivery priority score (0-100)."""
        score = self.CATEGORY_RULES.get(category, {}).get("priority", 1) * 10
        
        # Boost recent files
        age_hours = (time.time() - file_meta.get("mtime", 0)) / 3600
        if age_hours < 24:
            score += 20
        elif age_hours < 168:  # 1 week
            score += 10
        
        # Boost final/release versions
        hints = file_meta.get("content_hints", {})
        if hints.get("is_final"):
            score += 15
        if hints.get("has_version"):
            score += 5
        
        # Penalize drafts
        if hints.get("is_draft"):
            score -= 10
        
        # Size factor (prefer medium-sized files)
        size = file_meta.get("size", 0)
        if 1024 < size < 10 * 1024 * 1024:  # 1KB - 10MB
            score += 10
        elif size > 50 * 1024 * 1024:  # > 50MB
            score -= 20
        
        return max(0, min(100, score))


class BatchOptimizer:
    """Optimize file batching for efficient delivery."""
    
    def __init__(self, max_batch_size: int = 10, max_total_mb: float = 50.0):
        self.max_batch_size = max_batch_size
        self.max_total_bytes = int(max_total_mb * 1024 * 1024)
        self.delivery_history = []
    
    def optimize_batch(self, files: list, category_engine: FileCategorizationEngine) -> list:
        """Create optimized batch using greedy knapsack approach."""
        # Score all files
        scored_files = []
        for f in files:
            category = category_engine.categorize(f)
            priority = category_engine.get_priority_score(category, f)
            scored_files.append({
                **f,
                "category": category,
                "priority": priority
            })
        
        # Sort by priority (descending)
        scored_files.sort(key=lambda x: x["priority"], reverse=True)
        
        # Greedy selection
        batch = []
        total_size = 0
        
        for f in scored_files:
            if len(batch) >= self.max_batch_size:
                break
            
            if total_size + f["size"] > self.max_total_bytes:
                continue
            
            batch.append(f)
            total_size += f["size"]
        
        return batch
    
    def analyze_batch_composition(self, batch: list) -> dict:
        """Analyze batch composition and diversity."""
        categories = Counter(f["category"] for f in batch)
        extensions = Counter(f["ext"] for f in batch)
        
        total_size = sum(f["size"] for f in batch)
        avg_priority = sum(f["priority"] for f in batch) / len(batch) if batch else 0
        
        return {
            "file_count": len(batch),
            "total_size": total_size,
            "avg_priority": avg_priority,
            "category_distribution": dict(categories),
            "extension_distribution": dict(extensions),
            "diversity_score": len(categories) / len(batch) if batch else 0
        }


class FileMonitor:
    """Monitor and track filesystem changes."""
    
    def __init__(self, workspace: pathlib.Path):
        self.workspace = workspace
        self.known_files = {}
        self.change_log = []
    
    def scan(self, max_depth: int = 3) -> dict:
        """Scan workspace and detect changes."""
        current_files = {}
        new_files = []
        modified_files = []
        
        def scan_recursive(path: pathlib.Path, depth: int = 0):
            if depth > max_depth:
                return
            
            try:
                for item in path.iterdir():
                    if item.is_file():
                        # Skip system/hidden files
                        if item.name.startswith('.') or item.name.endswith('.pyc'):
                            continue
                        
                        try:
                            stat = item.stat()
                            file_id = str(item.relative_to(self.workspace))
                            
                            current_files[file_id] = {
                                "path": item,
                                "mtime": stat.st_mtime,
                                "size": stat.st_size
                            }
                            
                            # Detect changes
                            if file_id not in self.known_files:
                                new_files.append(file_id)
                            elif self.known_files[file_id]["mtime"] < stat.st_mtime:
                                modified_files.append(file_id)
                        
                        except Exception:
                            continue
                    
                    elif item.is_dir() and not item.name.startswith('.'):
                        scan_recursive(item, depth + 1)
            
            except Exception:
                pass
        
        scan_recursive(self.workspace)
        
        # Update known files
        self.known_files = current_files
        
        return {
            "total_files": len(current_files),
            "new_files": new_files,
            "modified_files": modified_files,
            "scan_time": datetime.now().isoformat()
        }


def format_size(size: int) -> str:
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def main() -> None:
    """Main loop - Intelligent file curation and batch optimization."""
    heartbeat_path_str = os.environ.get("PROTEA_HEARTBEAT")
    if not heartbeat_path_str:
        print("ERROR: PROTEA_HEARTBEAT not set", flush=True)
        return
    
    heartbeat_path = pathlib.Path(heartbeat_path_str)
    pid = os.getpid()
    
    try:
        heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
    except Exception as e:
        print(f"ERROR: Cannot write heartbeat: {e}", flush=True)
        return
    
    stop_event = Event()
    heartbeat_thread = Thread(
        target=heartbeat_loop,
        args=(heartbeat_path, pid, stop_event),
        daemon=True
    )
    heartbeat_thread.start()
    
    # Setup
    workspace = pathlib.Path(__file__).parent
    
    print(f"╔{'═' * 78}╗", flush=True)
    print(f"║{'🎯 Ring 2 Generation 407: Intelligent File Curator'.center(78)}║", flush=True)
    print(f"╚{'═' * 78}╝", flush=True)
    print(f"📍 PID: {pid}", flush=True)
    print(f"⏰ Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"🎯 Mission: Smart file discovery, categorization & batch optimization", flush=True)
    print(f"📂 Workspace: {workspace}", flush=True)
    
    # Initialize engines
    monitor = FileMonitor(workspace)
    category_engine = FileCategorizationEngine()
    batch_optimizer = BatchOptimizer(max_batch_size=10, max_total_mb=45.0)
    
    cycle_count = 0
    total_files_analyzed = 0
    total_batches_created = 0
    
    try:
        while True:
            print(f"\n{'━' * 80}", flush=True)
            print(f"🔄 CURATION CYCLE {cycle_count + 1}", flush=True)
            print(f"🕐 {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print(f"{'━' * 80}\n", flush=True)
            
            # Scan for changes
            print("🔍 Scanning workspace for files...", flush=True)
            scan_result = monitor.scan(max_depth=3)
            
            print(f"✅ Scan complete:", flush=True)
            print(f"   📊 Total files: {scan_result['total_files']}", flush=True)
            print(f"   🆕 New: {len(scan_result['new_files'])}", flush=True)
            print(f"   📝 Modified: {len(scan_result['modified_files'])}", flush=True)
            
            if scan_result['total_files'] == 0:
                print("📭 No files found", flush=True)
                time.sleep(60)
                continue
            
            # Extract metadata for all files
            print(f"\n📋 Extracting file metadata...", flush=True)
            all_files = []
            for file_id, file_info in monitor.known_files.items():
                meta = FileSignature.extract_metadata(file_info["path"])
                if "error" not in meta:
                    meta["relative_path"] = file_id
                    meta["file_hash"] = FileSignature.compute_hash(file_info["path"])
                    all_files.append(meta)
            
            total_files_analyzed += len(all_files)
            print(f"✅ Metadata extracted for {len(all_files)} files", flush=True)
            
            # Categorize files
            print(f"\n🏷️  Categorizing files...", flush=True)
            categorized = defaultdict(list)
            for f in all_files:
                category = category_engine.categorize(f)
                f["category"] = category
                categorized[category].append(f)
            
            print(f"\n{'─' * 80}", flush=True)
            print(f"{'CATEGORY':<15} {'COUNT':>8} {'TOTAL SIZE':>15} {'AVG SIZE':>15}", flush=True)
            print(f"{'─' * 80}", flush=True)
            
            for category in sorted(categorized.keys(), key=lambda c: len(categorized[c]), reverse=True):
                files = categorized[category]
                total_size = sum(f["size"] for f in files)
                avg_size = total_size / len(files) if files else 0
                
                print(f"{category:<15} {len(files):>8} {format_size(total_size):>15} {format_size(avg_size):>15}", flush=True)
            
            print(f"{'─' * 80}", flush=True)
            
            # Create optimized batch
            print(f"\n🎯 Creating optimized delivery batch...", flush=True)
            batch = batch_optimizer.optimize_batch(all_files, category_engine)
            
            if not batch:
                print("⚠️  No suitable files for batching", flush=True)
            else:
                total_batches_created += 1
                analysis = batch_optimizer.analyze_batch_composition(batch)
                
                print(f"\n{'═' * 80}", flush=True)
                print(f"📦 OPTIMIZED BATCH #{total_batches_created}", flush=True)
                print(f"{'═' * 80}", flush=True)
                print(f"📊 Files: {analysis['file_count']}", flush=True)
                print(f"💾 Total size: {format_size(analysis['total_size'])}", flush=True)
                print(f"⭐ Avg priority: {analysis['avg_priority']:.1f}/100", flush=True)
                print(f"🎨 Diversity: {analysis['diversity_score']:.2f}", flush=True)
                print(f"", flush=True)
                
                print(f"📋 Category distribution:", flush=True)
                for cat, count in sorted(analysis['category_distribution'].items(), key=lambda x: x[1], reverse=True):
                    print(f"   {cat:<15} {count:>3} files", flush=True)
                
                print(f"\n📄 Batch contents (sorted by priority):", flush=True)
                print(f"{'─' * 80}", flush=True)
                print(f"{'#':<3} {'PRIORITY':>8} {'CATEGORY':<12} {'SIZE':>10} {'FILE':<40}", flush=True)
                print(f"{'─' * 80}", flush=True)
                
                for idx, f in enumerate(batch[:15], 1):
                    name = f["name"][:38] + "..." if len(f["name"]) > 40 else f["name"]
                    print(f"{idx:<3} {f['priority']:>8.1f} {f['category']:<12} {format_size(f['size']):>10} {name:<40}", flush=True)
                
                if len(batch) > 15:
                    print(f"... and {len(batch) - 15} more files", flush=True)
                
                print(f"{'─' * 80}", flush=True)
                
                # Pattern analysis
                print(f"\n🔍 Pattern analysis:", flush=True)
                date_files = sum(1 for f in batch if f.get("content_hints", {}).get("has_date"))
                version_files = sum(1 for f in batch if f.get("content_hints", {}).get("has_version"))
                final_files = sum(1 for f in batch if f.get("content_hints", {}).get("is_final"))
                draft_files = sum(1 for f in batch if f.get("content_hints", {}).get("is_draft"))
                
                print(f"   📅 Files with dates: {date_files}", flush=True)
                print(f"   🔢 Versioned files: {version_files}", flush=True)
                print(f"   ✅ Final versions: {final_files}", flush=True)
                print(f"   📝 Drafts: {draft_files}", flush=True)
                
                # Save batch manifest
                manifest_path = workspace / f"batch_manifest_{cycle_count + 1}.json"
                manifest = {
                    "cycle": cycle_count + 1,
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis,
                    "files": [
                        {
                            "name": f["name"],
                            "category": f["category"],
                            "priority": f["priority"],
                            "size": f["size"],
                            "path": str(f["relative_path"])
                        }
                        for f in batch
                    ]
                }
                
                try:
                    with open(manifest_path, 'w', encoding='utf-8') as mf:
                        json.dump(manifest, mf, indent=2, ensure_ascii=False)
                    print(f"\n💾 Manifest saved: {manifest_path.name}", flush=True)
                except Exception as e:
                    print(f"⚠️  Could not save manifest: {e}", flush=True)
            
            # Summary
            print(f"\n{'═' * 80}", flush=True)
            print(f"📊 SESSION SUMMARY", flush=True)
            print(f"{'═' * 80}", flush=True)
            print(f"🔄 Cycles completed: {cycle_count + 1}", flush=True)
            print(f"📁 Total files analyzed: {total_files_analyzed}", flush=True)
            print(f"📦 Batches created: {total_batches_created}", flush=True)
            print(f"{'═' * 80}", flush=True)
            
            cycle_count += 1
            
            print(f"\n⏳ Next scan in 90 seconds...\n", flush=True)
            time.sleep(90)
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted - shutting down...", flush=True)
    except Exception as e:
        print(f"\n❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=5)
        try:
            heartbeat_path.unlink(missing_ok=True)
        except Exception:
            pass
        
        print(f"\n{'═' * 80}", flush=True)
        print(f"🏁 File Curator Shutdown", flush=True)
        print(f"📊 Cycles: {cycle_count}", flush=True)
        print(f"📁 Files analyzed: {total_files_analyzed}", flush=True)
        print(f"📦 Batches created: {total_batches_created}", flush=True)
        print(f"{'═' * 80}", flush=True)


if __name__ == "__main__":
    main()