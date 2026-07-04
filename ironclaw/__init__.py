"""Ironclaw — a multi-agent research institute.

Layers (each talks only to its direct superior and subordinate):
    PI  ->  Senior Researcher  ->  PhD          (Postdoc reviews the PhD)
Sibling service: the Infrastructure Manager, custodian of skills and resources.

This package currently implements the PhD vertical slice: the executor that does
~90% of the real work.
"""

__version__ = "0.0.1"
