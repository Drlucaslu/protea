"""Automatic skill crystallization from stable Ring 2 code patterns.

Detects high-performing, stable code modules across generations and
automatically extracts them as reusable skills.

Pure stdlib.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("protea.crystallizer")


@dataclass
class CodeModule:
    """A stable code module detected across generations."""
    name: str
    code: str
    type: str  # "function", "class", or "standalone"
    stability_score: float  # 0.0-1.0
    first_seen: int  # generation
    last_seen: int
    occurrences: int


class AutoCrystallizer:
    """Automatically extracts stable, high-value code into Skills."""
    
    def __init__(
        self,
        skills_dir: pathlib.Path,
        min_stability: float = 0.80,
        min_score: float = 0.85,
        min_occurrences: int = 5,
    ):
        """
        Args:
            skills_dir: Directory to save crystallized skills
            min_stability: Minimum stability score (0.0-1.0)
            min_score: Minimum fitness score to consider
            min_occurrences: Minimum generations a pattern must appear
        """
        self.skills_dir = skills_dir
        self.min_stability = min_stability
        self.min_score = min_score
        self.min_occurrences = min_occurrences
        
        # Track code patterns across generations
        self.module_history: dict[str, CodeModule] = {}
    
    def analyze_generation(
        self,
        generation: int,
        source_code: str,
        fitness_score: float,
    ) -> None:
        """Analyze a generation's code and update module tracking.
        
        Args:
            generation: Generation number
            source_code: Ring 2 source code
            fitness_score: Fitness score for this generation
        """
        if fitness_score < self.min_score:
            return  # Only track high-performing code
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return
        
        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._track_module(node, generation, "function", source_code)
            elif isinstance(node, ast.ClassDef):
                self._track_module(node, generation, "class", source_code)
    
    def _track_module(
        self,
        node: ast.FunctionDef | ast.ClassDef,
        generation: int,
        module_type: str,
        full_source: str,
    ) -> None:
        """Track a function or class across generations."""
        # Skip heartbeat-related code (infrastructure, not reusable)
        if "heartbeat" in node.name.lower():
            return
        if node.name == "main":
            return
        
        # Extract the module's source code
        try:
            code = ast.get_source_segment(full_source, node)
            if not code:
                return
        except (ValueError, TypeError):
            return
        
        # Generate a stable signature (hash of normalized code)
        normalized = self._normalize_code(code)
        signature = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        module_id = f"{node.name}_{signature}"
        
        # Update tracking
        if module_id in self.module_history:
            module = self.module_history[module_id]
            module.last_seen = generation
            module.occurrences += 1
            module.stability_score = self._compute_stability(module)
        else:
            self.module_history[module_id] = CodeModule(
                name=node.name,
                code=code,
                type=module_type,
                stability_score=0.0,
                first_seen=generation,
                last_seen=generation,
                occurrences=1,
            )
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison (remove comments, extra whitespace)."""
        # Remove comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        # Remove empty lines
        lines = [ln.strip() for ln in code.split("\n") if ln.strip()]
        return "\n".join(lines)
    
    def _compute_stability(self, module: CodeModule) -> float:
        """Compute stability score based on occurrence pattern."""
        span = module.last_seen - module.first_seen + 1
        if span == 0:
            return 0.0
        # Stability = occurrences / span (how consistently it appears)
        return min(module.occurrences / span, 1.0)
    
    def get_crystallization_candidates(self) -> list[CodeModule]:
        """Return modules ready for crystallization."""
        candidates = []
        for module in self.module_history.values():
            if (
                module.stability_score >= self.min_stability
                and module.occurrences >= self.min_occurrences
            ):
                candidates.append(module)
        
        # Sort by stability (best first)
        candidates.sort(key=lambda m: m.stability_score, reverse=True)
        return candidates
    
    def crystallize_module(
        self,
        module: CodeModule,
        description: str = "",
    ) -> pathlib.Path:
        """Create a Skill from a stable module.
        
        Args:
            module: CodeModule to crystallize
            description: Optional description (auto-generated if empty)
        
        Returns:
            Path to the created skill file
        """
        # Generate skill name
        skill_name = f"auto_{module.name.lower()}"
        skill_path = self.skills_dir / skill_name / "main.py"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-generate description if not provided
        if not description:
            description = (
                f"Auto-crystallized {module.type} from Ring 2 evolution. "
                f"Appeared in {module.occurrences} generations "
                f"(gen {module.first_seen}-{module.last_seen}), "
                f"stability score: {module.stability_score:.2f}"
            )
        
        # Generate skill code
        skill_code = self._generate_skill_code(module, description)
        
        # Write skill
        skill_path.write_text(skill_code)
        
        # Write metadata
        metadata = {
            "name": skill_name,
            "description": description,
            "type": module.type,
            "source": "auto_crystallization",
            "stability_score": module.stability_score,
            "generations": f"{module.first_seen}-{module.last_seen}",
            "occurrences": module.occurrences,
        }
        metadata_path = skill_path.parent / "skill.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
        log.info(
            f"Crystallized skill '{skill_name}' from module '{module.name}' "
            f"(stability: {module.stability_score:.2f})"
        )
        
        return skill_path
    
    def _generate_skill_code(self, module: CodeModule, description: str) -> str:
        """Generate complete skill code from a module."""
        return f'''#!/usr/bin/env python3
"""
{description}
"""

import os
import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler


class _ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


# ============= CRYSTALLIZED MODULE =============

{module.code}


# ============= HTTP API SERVER =============

class SkillHandler(BaseHTTPRequestHandler):
    """Simple HTTP API for the crystallized skill."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
            <html>
            <body>
                <h1>Auto-Crystallized Skill: {module.name}</h1>
                <p>{description}</p>
                <p><a href="/api/info">Skill Info</a></p>
            </body>
            </html>
            """)
        elif self.path == "/api/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            info = {{
                "skill": "{module.name}",
                "type": "{module.type}",
                "status": "active",
            }}
            self.wfile.write(json.dumps(info).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress request logging."""
        pass


def main():
    """Run the skill as an HTTP service."""
    port = int(os.environ.get("SKILL_PORT", 8800))
    server = _ReusableHTTPServer(("0.0.0.0", port), SkillHandler)
    print(f"Skill '{module.name}' running on http://localhost:{{port}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
'''
    
    def auto_crystallize_check(self, generation: int) -> list[pathlib.Path]:
        """Check if any modules are ready for crystallization.
        
        Should be called periodically (e.g., every 10 generations).
        
        Args:
            generation: Current generation number
        
        Returns:
            List of paths to newly created skills
        """
        candidates = self.get_crystallization_candidates()
        
        if not candidates:
            log.debug(f"Gen {generation}: No crystallization candidates found")
            return []
        
        # Check if already crystallized
        existing_skills = {p.parent.name for p in self.skills_dir.glob("auto_*/main.py")}
        
        new_skills = []
        for module in candidates:
            skill_name = f"auto_{module.name.lower()}"
            if skill_name in existing_skills:
                continue  # Already crystallized
            
            # Crystallize!
            skill_path = self.crystallize_module(module)
            new_skills.append(skill_path)
            log.info(
                f"🔮 Gen {generation}: Auto-crystallized skill '{skill_name}' "
                f"(stability: {module.stability_score:.2f}, "
                f"occurrences: {module.occurrences})"
            )
        
        return new_skills
    
    def get_statistics(self) -> dict[str, Any]:
        """Get crystallization statistics."""
        candidates = self.get_crystallization_candidates()
        return {
            "total_modules_tracked": len(self.module_history),
            "crystallization_candidates": len(candidates),
            "top_candidates": [
                {
                    "name": m.name,
                    "stability": m.stability_score,
                    "occurrences": m.occurrences,
                }
                for m in candidates[:5]
            ],
        }
