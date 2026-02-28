"""Tests for GenePool.diff_capabilities."""

from __future__ import annotations

from ring0.gene_pool import GenePool


class TestDiffCapabilities:
    def test_detects_new_class(self):
        old = "class Foo: pass\ndef bar(): pass"
        new = "class Foo: pass\ndef bar(): pass\nclass Baz: pass"
        diff = GenePool.diff_capabilities(old, new)
        assert diff == ["Baz"]

    def test_detects_new_function(self):
        old = "def foo(): pass"
        new = "def foo(): pass\ndef bar(): pass"
        diff = GenePool.diff_capabilities(old, new)
        assert diff == ["bar"]

    def test_no_diff_when_identical(self):
        code = "class Foo: pass\ndef bar(): pass"
        diff = GenePool.diff_capabilities(code, code)
        assert diff == []

    def test_empty_old(self):
        new = "class Alpha: pass\ndef beta(): pass"
        diff = GenePool.diff_capabilities("", new)
        assert set(diff) == {"Alpha", "beta"}

    def test_empty_new(self):
        old = "class Alpha: pass"
        diff = GenePool.diff_capabilities(old, "")
        assert diff == []

    def test_multiple_new_caps(self):
        old = "class A: pass"
        new = "class A: pass\nclass B: pass\ndef c(): pass\ndef d(): pass"
        diff = GenePool.diff_capabilities(old, new)
        assert set(diff) == {"B", "c", "d"}

    def test_ignores_removed_caps(self):
        old = "class A: pass\nclass B: pass"
        new = "class A: pass"
        diff = GenePool.diff_capabilities(old, new)
        assert diff == []

    def test_handles_methods_in_summary(self):
        old = "class Foo:\n  def method1(): pass"
        new = "class Foo:\n  def method1(): pass\n  def method2(): pass\nclass Bar: pass"
        diff = GenePool.diff_capabilities(old, new)
        assert set(diff) == {"method2", "Bar"}
