#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class CoverageStats:
    name: str
    covered: int
    total: int

    @property
    def percent(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.covered / self.total) * 100.0


def parse_python_coverage(path: Path) -> CoverageStats:
    root = ET.parse(path).getroot()
    total = int(root.attrib["lines-valid"])
    covered = int(root.attrib["lines-covered"])
    return CoverageStats(name="Python", covered=covered, total=total)


def parse_rust_coverage(path: Path) -> CoverageStats:
    covered = 0
    total = 0

    for line in path.read_text().splitlines():
        if line.startswith("LH:"):
            covered += int(line[3:])
        elif line.startswith("LF:"):
            total += int(line[3:])

    return CoverageStats(name="Rust", covered=covered, total=total)


def build_summary(python_stats: CoverageStats, rust_stats: CoverageStats) -> str:
    rows = "\n".join(
        f"| {stats.name} | {stats.covered} | {stats.total} | {stats.percent:.2f}% |"
        for stats in (python_stats, rust_stats)
    )
    return "\n".join(
        [
            "## Coverage Report",
            "",
            "| Scope | Covered Lines | Total Lines | Line Coverage |",
            "| --- | ---: | ---: | ---: |",
            rows,
            "",
            "Artifacts:",
            "- Python XML/HTML coverage: `atompack-py/coverage/`",
            "- Rust LCOV coverage: `coverage/rust.lcov`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a markdown coverage summary.")
    parser.add_argument("--python-xml", type=Path, required=True)
    parser.add_argument("--rust-lcov", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    python_stats = parse_python_coverage(args.python_xml)
    rust_stats = parse_rust_coverage(args.rust_lcov)
    summary = build_summary(python_stats, rust_stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
