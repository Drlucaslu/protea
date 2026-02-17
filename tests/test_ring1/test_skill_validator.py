"""Tests for ring1.skill_validator — security validation for skills."""

from ring1.skill_validator import validate_skill, ValidationResult


class TestValidateSkill:
    """validate_skill() should detect dangerous patterns."""

    def test_safe_http_server_skill(self):
        code = '''
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

HTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
'''
        result = validate_skill(code)
        assert result.safe is True
        assert result.errors == []

    def test_rejects_os_system(self):
        result = validate_skill('import os\nos.system("rm -rf /")')
        assert result.safe is False
        assert any("os.system" in e for e in result.errors)

    def test_rejects_subprocess(self):
        result = validate_skill('import subprocess\nsubprocess.run(["ls"])')
        assert result.safe is False
        assert any("subprocess" in e for e in result.errors)

    def test_rejects_eval(self):
        result = validate_skill('x = eval(input())')
        assert result.safe is False
        assert any("eval" in e for e in result.errors)

    def test_rejects_exec(self):
        result = validate_skill('exec("print(1)")')
        assert result.safe is False
        assert any("exec" in e for e in result.errors)

    def test_rejects_shutil_rmtree(self):
        result = validate_skill('import shutil\nshutil.rmtree("/tmp/data")')
        assert result.safe is False
        assert any("shutil.rmtree" in e for e in result.errors)

    def test_rejects_dunder_import(self):
        result = validate_skill('m = __import__("os")')
        assert result.safe is False
        assert any("__import__" in e for e in result.errors)

    def test_rejects_sensitive_file_access(self):
        result = validate_skill('open("/etc/passwd")')
        assert result.safe is False
        assert any("sensitive" in e for e in result.errors)

    def test_rejects_raw_socket(self):
        result = validate_skill('import socket\ns = socket.socket()')
        assert result.safe is False
        assert any("socket" in e for e in result.errors)

    def test_rejects_smtp(self):
        result = validate_skill('import smtplib\nsmtplib.SMTP("mail.com")')
        assert result.safe is False
        assert any("SMTP" in e for e in result.errors)

    def test_rejects_ctypes(self):
        result = validate_skill('import ctypes\nctypes.cdll.LoadLibrary("libc.so")')
        assert result.safe is False
        assert any("ctypes" in e for e in result.errors)

    def test_rejects_os_remove(self):
        result = validate_skill('import os\nos.remove("/tmp/file")')
        assert result.safe is False
        assert any("os.remove" in e for e in result.errors)

    def test_warns_on_file_write(self):
        result = validate_skill('f = open("output.txt", "w")')
        assert result.safe is True  # warnings don't block
        assert any("writing" in w for w in result.warnings)

    def test_warns_on_urllib(self):
        result = validate_skill('import urllib.request\nurllib.request.urlopen("http://example.com")')
        assert result.safe is True
        assert any("HTTP" in w for w in result.warnings)

    def test_warns_on_pickle(self):
        result = validate_skill('import pickle\npickle.loads(data)')
        assert result.safe is True
        assert any("pickle" in w for w in result.warnings)

    def test_empty_source_rejected(self):
        result = validate_skill("")
        assert result.safe is False
        assert any("empty" in e for e in result.errors)

    def test_whitespace_only_rejected(self):
        result = validate_skill("   \n\n  ")
        assert result.safe is False

    def test_multiple_violations(self):
        code = 'os.system("bad")\neval(x)\nexec(y)'
        result = validate_skill(code)
        assert result.safe is False
        assert len(result.errors) >= 3

    def test_validation_result_repr(self):
        result = ValidationResult()
        assert "SAFE" in repr(result)
        result.safe = False
        result.errors = ["test"]
        assert "BLOCKED" in repr(result)
