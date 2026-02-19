#!/usr/bin/env python3

import os
import pathlib
import time
import json
import sys
from threading import Thread, Event
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import signal
import re
from collections import defaultdict, Counter
import hashlib

HEARTBEAT_INTERVAL = 2

def heartbeat_loop(heartbeat_path: pathlib.Path, pid: int, stop_event: Event) -> None:
    while not stop_event.is_set():
        try:
            heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
        except Exception:
            pass
        time.sleep(HEARTBEAT_INTERVAL)

class LogPattern:
    """Represents a detected pattern in log files"""
    def __init__(self, pattern_type: str, pattern: str, count: int, severity: str):
        self.pattern_type = pattern_type
        self.pattern = pattern
        self.count = count
        self.severity = severity
        self.samples: List[str] = []
        
    def add_sample(self, line: str) -> None:
        if len(self.samples) < 3:
            self.samples.append(line[:200])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.pattern_type,
            'pattern': self.pattern,
            'count': self.count,
            'severity': self.severity,
            'samples': self.samples
        }

class LogAnalyzer:
    """Analyzes log files for errors, patterns, and anomalies"""
    
    ERROR_PATTERNS = [
        (r'\bERROR\b', 'ERROR', 'high'),
        (r'\bFAIL(ED)?\b', 'FAILURE', 'high'),
        (r'\bCRITICAL\b', 'CRITICAL', 'critical'),
        (r'\bWARN(ING)?\b', 'WARNING', 'medium'),
        (r'\bEXCEPTION\b', 'EXCEPTION', 'high'),
        (r'Traceback \(most recent call last\)', 'TRACEBACK', 'high'),
        (r'\btimeout\b', 'TIMEOUT', 'medium'),
        (r'\bconnection (refused|reset|closed)\b', 'CONNECTION', 'medium'),
        (r'\b(404|500|502|503)\b', 'HTTP_ERROR', 'medium'),
        (r'\bpermission denied\b', 'PERMISSION', 'medium'),
        (r'\bno such file\b', 'FILE_NOT_FOUND', 'low'),
    ]
    
    def __init__(self, config_path: Optional[pathlib.Path] = None):
        self.config = self._load_config(config_path)
        self.patterns_found: Dict[str, LogPattern] = {}
        self.line_count = 0
        self.error_count = 0
        
    def _load_config(self, config_path: Optional[pathlib.Path]) -> Dict[str, Any]:
        default_config = {
            'log_paths': [
                '/var/log/system.log',
                str(pathlib.Path.home() / 'Library' / 'Logs'),
                '/tmp/*.log'
            ],
            'max_lines_per_file': 10000,
            'time_window_hours': 24,
            'anomaly_threshold': 10,
            'custom_patterns': []
        }
        
        if config_path and config_path.exists():
            try:
                with config_path.open('r') as f:
                    return {**default_config, **json.load(f)}
            except:
                pass
        
        return default_config
    
    def find_log_files(self) -> List[pathlib.Path]:
        """Find all readable log files"""
        log_files = []
        
        for log_spec in self.config['log_paths']:
            try:
                path = pathlib.Path(log_spec).expanduser()
                
                if path.is_file() and path.suffix in ['.log', '.txt']:
                    if os.access(path, os.R_OK):
                        log_files.append(path)
                elif path.is_dir():
                    for log_file in path.rglob('*.log'):
                        if os.access(log_file, os.R_OK):
                            log_files.append(log_file)
                            if len(log_files) >= 20:
                                break
            except (PermissionError, OSError):
                continue
        
        return log_files[:20]
    
    def analyze_file(self, file_path: pathlib.Path) -> Dict[str, Any]:
        """Analyze a single log file"""
        stats = {
            'path': str(file_path),
            'size_bytes': 0,
            'lines_analyzed': 0,
            'errors_found': 0,
            'timestamps': [],
            'patterns': {}
        }
        
        try:
            stats['size_bytes'] = file_path.stat().st_size
            
            max_lines = self.config['max_lines_per_file']
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    
                    self.line_count += 1
                    stats['lines_analyzed'] += 1
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    timestamp = self._extract_timestamp(line)
                    if timestamp:
                        stats['timestamps'].append(timestamp)
                    
                    for pattern_re, pattern_name, severity in self.ERROR_PATTERNS:
                        if re.search(pattern_re, line, re.IGNORECASE):
                            self.error_count += 1
                            stats['errors_found'] += 1
                            
                            key = f"{pattern_name}_{severity}"
                            if key not in self.patterns_found:
                                self.patterns_found[key] = LogPattern(
                                    pattern_name, pattern_re, 0, severity
                                )
                            
                            self.patterns_found[key].count += 1
                            self.patterns_found[key].add_sample(line)
                            
                            stats['patterns'][pattern_name] = stats['patterns'].get(pattern_name, 0) + 1
        
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        """Extract ISO timestamp or common log timestamp formats"""
        patterns = [
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
            r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',
            r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(0)
        
        return None
    
    def detect_anomalies(self, file_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalous patterns based on frequency"""
        anomalies = []
        
        all_patterns = Counter()
        for stats in file_stats:
            for pattern, count in stats.get('patterns', {}).items():
                all_patterns[pattern] += count
        
        threshold = self.config['anomaly_threshold']
        for pattern, count in all_patterns.most_common():
            if count >= threshold:
                anomalies.append({
                    'pattern': pattern,
                    'occurrences': count,
                    'severity': 'high' if count >= threshold * 3 else 'medium',
                    'message': f'{pattern} occurred {count} times (threshold: {threshold})'
                })
        
        return anomalies
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        print("🔍 Discovering log files...", flush=True)
        log_files = self.find_log_files()
        
        if not log_files:
            print("⚠️  No readable log files found", flush=True)
            print("💡 Generating synthetic demo data...", flush=True)
            return self._generate_demo_report()
        
        print(f"📁 Found {len(log_files)} log files\n", flush=True)
        
        file_stats = []
        for log_file in log_files:
            print(f"📄 Analyzing: {log_file.name}", flush=True)
            stats = self.analyze_file(log_file)
            file_stats.append(stats)
        
        anomalies = self.detect_anomalies(file_stats)
        
        patterns_by_severity = defaultdict(list)
        for pattern in self.patterns_found.values():
            patterns_by_severity[pattern.severity].append(pattern)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'files_analyzed': len(log_files),
                'total_lines': self.line_count,
                'total_errors': self.error_count,
                'unique_patterns': len(self.patterns_found),
                'anomalies_detected': len(anomalies)
            },
            'file_statistics': file_stats,
            'patterns': {
                severity: [p.to_dict() for p in patterns]
                for severity, patterns in patterns_by_severity.items()
            },
            'anomalies': anomalies,
            'recommendations': self._generate_recommendations(anomalies)
        }
    
    def _generate_demo_report(self) -> Dict[str, Any]:
        """Generate demo report when no logs are accessible"""
        demo_patterns = [
            LogPattern('CONNECTION', r'connection refused', 45, 'medium'),
            LogPattern('TIMEOUT', r'timeout', 23, 'medium'),
            LogPattern('ERROR', r'ERROR', 67, 'high'),
            LogPattern('WARNING', r'WARNING', 124, 'medium'),
        ]
        
        for pattern in demo_patterns:
            pattern.add_sample(f'2024-01-15 10:23:45 {pattern.pattern.upper()}: Sample log entry')
            pattern.add_sample(f'2024-01-15 11:34:12 {pattern.pattern.upper()}: Another occurrence')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'mode': 'demo',
            'summary': {
                'files_analyzed': 0,
                'total_lines': 5420,
                'total_errors': 259,
                'unique_patterns': 4,
                'anomalies_detected': 2
            },
            'file_statistics': [],
            'patterns': {
                'high': [demo_patterns[2].to_dict()],
                'medium': [p.to_dict() for p in [demo_patterns[0], demo_patterns[1], demo_patterns[3]]]
            },
            'anomalies': [
                {'pattern': 'WARNING', 'occurrences': 124, 'severity': 'medium', 'message': 'WARNING occurred 124 times (threshold: 10)'},
                {'pattern': 'ERROR', 'occurrences': 67, 'severity': 'high', 'message': 'ERROR occurred 67 times (threshold: 10)'}
            ],
            'recommendations': [
                'Investigate high-frequency WARNING patterns',
                'Review ERROR logs for recurring issues',
                'Consider implementing rate limiting for connection errors'
            ]
        }
    
    def _generate_recommendations(self, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations"""
        recs = []
        
        for anomaly in anomalies[:5]:
            pattern = anomaly['pattern']
            count = anomaly['occurrences']
            
            if 'CONNECTION' in pattern:
                recs.append(f'Review network connectivity — {count} connection errors detected')
            elif 'TIMEOUT' in pattern:
                recs.append(f'Optimize timeout settings — {count} timeout events found')
            elif 'ERROR' in pattern or 'CRITICAL' in pattern:
                recs.append(f'Prioritize fixing {pattern} — {count} occurrences need attention')
            elif 'WARNING' in pattern and count > 50:
                recs.append(f'Investigate {count} warnings — may indicate underlying issue')
        
        if not recs:
            recs.append('No critical anomalies detected — system appears healthy')
        
        return recs

def main() -> None:
    heartbeat_path_str = os.environ.get("PROTEA_HEARTBEAT")
    if not heartbeat_path_str:
        print("ERROR: PROTEA_HEARTBEAT not set", flush=True)
        return

    heartbeat_path = pathlib.Path(heartbeat_path_str)
    pid = os.getpid()

    try:
        heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
    except Exception as e:
        print(f"ERROR: Heartbeat failed: {e}", flush=True)
        return

    stop_event = Event()

    def signal_handler(signum, frame):
        print(f"\nSignal {signum} received, shutting down", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    heartbeat_thread = Thread(target=heartbeat_loop, args=(heartbeat_path, pid, stop_event), daemon=True)
    heartbeat_thread.start()

    print("="*80, flush=True)
    print("PROTEA LOG ANALYZER", flush=True)
    print("Intelligent log file analysis with anomaly detection", flush=True)
    print("="*80, flush=True)
    print(f"PID: {pid} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    config_path = pathlib.Path.home() / '.protea' / 'log_analyzer_config.json'
    analyzer = LogAnalyzer(config_path)

    try:
        print(f"Configuration:\n{json.dumps(analyzer.config, indent=2)}\n", flush=True)

        report = analyzer.generate_report()

        print("\n" + "─"*80, flush=True)
        print("ANALYSIS SUMMARY", flush=True)
        print("─"*80, flush=True)
        summary = report['summary']
        print(f"Files analyzed: {summary['files_analyzed']}", flush=True)
        print(f"Total lines: {summary['total_lines']:,}", flush=True)
        print(f"Errors found: {summary['total_errors']:,}", flush=True)
        print(f"Unique patterns: {summary['unique_patterns']}", flush=True)
        print(f"Anomalies: {summary['anomalies_detected']}", flush=True)

        if report.get('mode') == 'demo':
            print("\n⚠️  Demo mode: No accessible logs found, showing sample analysis", flush=True)

        print("\n" + "─"*80, flush=True)
        print("ERROR PATTERNS BY SEVERITY", flush=True)
        print("─"*80, flush=True)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in report['patterns']:
                print(f"\n{severity.upper()}:", flush=True)
                for pattern in report['patterns'][severity]:
                    print(f"  • {pattern['type']}: {pattern['count']} occurrences", flush=True)
                    if pattern['samples']:
                        print(f"    Sample: {pattern['samples'][0][:80]}...", flush=True)

        if report['anomalies']:
            print("\n" + "─"*80, flush=True)
            print("⚠️  ANOMALIES DETECTED", flush=True)
            print("─"*80, flush=True)
            for i, anomaly in enumerate(report['anomalies'], 1):
                severity_icon = "🔴" if anomaly['severity'] == 'high' else "🟡"
                print(f"{i}. {severity_icon} {anomaly['message']}", flush=True)

        if report['recommendations']:
            print("\n" + "─"*80, flush=True)
            print("💡 RECOMMENDATIONS", flush=True)
            print("─"*80, flush=True)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}", flush=True)

        if report['file_statistics']:
            print("\n" + "─"*80, flush=True)
            print("FILE DETAILS", flush=True)
            print("─"*80, flush=True)
            for stats in report['file_statistics'][:10]:
                print(f"\n📄 {pathlib.Path(stats['path']).name}", flush=True)
                print(f"   Size: {stats['size_bytes']:,} bytes", flush=True)
                print(f"   Lines: {stats['lines_analyzed']:,}", flush=True)
                print(f"   Errors: {stats['errors_found']}", flush=True)

        print("\n" + "="*80, flush=True)
        print("JSON OUTPUT", flush=True)
        print("="*80, flush=True)
        print(json.dumps(report, indent=2), flush=True)

        print("\n" + "="*80, flush=True)
        print(f"Analysis complete. Processed {summary['total_lines']:,} lines.", flush=True)
        print("="*80, flush=True)

        while not stop_event.is_set():
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)
        stop_event.set()
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=5)
        try:
            heartbeat_path.unlink(missing_ok=True)
        except:
            pass
        print(f"\nSession ended", flush=True)

if __name__ == "__main__":
    main()