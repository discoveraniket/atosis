#!/usr/bin/env python3
"""
Converter for H2 Spatial Tax Pilot (S1_prereg_pilot.md §6)

Converts Condition B/C brace-syntax generations to Python for scoring.
Implements the exact rules from §6 "Conversion to Python (B and C)".
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ConversionError(Enum):
    """Error codes from §6 conversion."""
    E_COLON_BRACE = "E_COLON_BRACE"      # Python-style header with ':' before '{'
    E_HEADER_SYNTAX = "E_HEADER_SYNTAX"  # Python-style header without '{' in B/C
    E_UNCLOSED = "E_UNCLOSED"            # Depth > 0 at end of extracted text
    E_INDENT = "E_INDENT"                # Indentation error in Condition A (not used here)


@dataclass
class ConversionResult:
    """Result of converting a B/C generation to Python."""
    python_code: str
    errors: List[ConversionError]
    depth_at_end: int
    colon_brace_flag: bool  # True if any line had ':' before '{'


def strip_trailing_comment(line: str) -> str:
    """Remove trailing '#' comment from a line."""
    if '#' in line:
        return line[:line.index('#')]
    return line


def classify_line(line: str) -> tuple[str, Optional[str]]:
    """
    Classify a line after stripping comment and whitespace.
    Returns (line_type, content) where line_type is one of:
    - 'empty'
    - 'close' (exactly '}')
    - 'open' (ends with '{')
    - 'statement'
    """
    stripped = line.strip()
    if not stripped:
        return 'empty', None
    if stripped == '}':
        return 'close', None
    if stripped.endswith('{'):
        header = stripped[:-1].rstrip()
        return 'open', header
    return 'statement', stripped


def convert_brace_to_python(generated_text: str) -> ConversionResult:
    """
    Convert Condition B/C generated text to Python per §6 rules.
    
    Rules:
    - Prompt supplies opening '{', so depth starts at 1
    - Each line classified after removing trailing '#' comment and whitespace
    - Empty line: skipped
    - Line exactly '}': closes one block (depth - 1)
    - Line ending with '{': opens block; header is text before '{'
      - Single ':' immediately before '{' removed, flag E_COLON_BRACE
    - Any other line: statement; braces anywhere else left in place
    - Output re-indented by depth (4 spaces/level); original indentation ignored
    - Python-style header without '{' -> E_HEADER_SYNTAX
    - Depth > 0 at end -> E_UNCLOSED
    """
    lines = generated_text.splitlines()
    errors = []
    colon_brace_flag = False
    depth = 0  # Will be incremented when we see the function signature's '{'
    output_lines = []
    seen_function_signature = False
    
    for line in lines:
        # Remove trailing comment
        line_no_comment = strip_trailing_comment(line)
        line_type, content = classify_line(line_no_comment)
        
        if line_type == 'empty':
            continue
        
        # Handle function signature specially - it's the first non-empty line
        if not seen_function_signature and line_type == 'open' and content.startswith('def process_data(data)'):
            seen_function_signature = True
            # Check for ':' before '{'
            if content.endswith(':'):
                content = content[:-1].rstrip()
                colon_brace_flag = True
                errors.append(ConversionError.E_COLON_BRACE)
            output_lines.append(f"{content}:")
            depth = 1  # Function body starts at depth 1
            continue
        
        elif line_type == 'close':
            depth -= 1
            if depth < 0:
                depth = 0
                    # Do NOT output '}' - Python uses indentation, not braces
                    # Just adjust depth; the next line will use the new depth
        
        elif line_type == 'open':
            header = content
            # Check for ':' immediately before '{'
            if header.endswith(':'):
                header = header[:-1].rstrip()
                colon_brace_flag = True
                errors.append(ConversionError.E_COLON_BRACE)
            
            indent = '    ' * depth
            output_lines.append(f"{indent}{header}:")
            depth += 1
        
        elif line_type == 'statement':
            # Check for braces in statement (not at end)
            if '{' in content or '}' in content:
                # Braces in statement -> will cause syntax error in Python
                pass  # Leave as-is, Python parser will reject
            
            # Check for Python-style header in B/C (block header without '{}')
            # This is a line that looks like a block header but doesn't end with '{}'
            # and is not a simple assignment/return
            if (content.startswith('if ') or content.startswith('for ')) and content.endswith(':'):
                errors.append(ConversionError.E_HEADER_SYNTAX)
            
            indent = '    ' * depth
            output_lines.append(f"{indent}{content}")
    
    # Check for unclosed blocks (depth should be 0 after function closes)
    if depth > 0:
        errors.append(ConversionError.E_UNCLOSED)
    
    python_code = '\n'.join(output_lines)
    
    return ConversionResult(
        python_code=python_code,
        errors=errors,
        depth_at_end=depth,
        colon_brace_flag=colon_brace_flag
    )


def python_to_condition_b(canonical_python: str) -> str:
    """
    Convert canonical Python to Condition B (braces with indentation per §2).
    """
    lines = canonical_python.splitlines()
    output = []
    depth = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        indent = len(line) - len(line.lstrip())
        indent_level = indent // 4

        # Close blocks if we have dedented
        while indent_level < depth:
            depth -= 1
            close_indent = '    ' * depth
            output.append(f"{close_indent}}}")

        if stripped.startswith('def process_data(data):'):
            output.append('def process_data(data) {')
            depth = 1
        elif stripped.endswith(':'):
            header = stripped[:-1]  # Remove trailing ':'
            header_indent = '    ' * depth
            output.append(f"{header_indent}{header} {{")
            depth += 1
        elif stripped == 'return total':
            return_indent = '    ' * depth
            output.append(f"{return_indent}return total")
        else:
            stmt_indent = '    ' * depth
            output.append(f"{stmt_indent}{stripped}")

    while depth > 0:
        depth -= 1
        close_indent = '    ' * depth
        output.append(f"{close_indent}}}")

    return '\n'.join(output)


def condition_b_to_condition_c(b_code: str) -> str:
    """
    Convert Condition B to Condition C (braces, no indentation per §2 & §8).
    Condition C equals Condition B with leading whitespace removed from every line.
    """
    return '\n'.join(line.lstrip() for line in b_code.splitlines())


def python_to_condition_c(canonical_python: str) -> str:
    """Convert canonical Python directly to Condition C."""
    return condition_b_to_condition_c(python_to_condition_b(canonical_python))


# Alias for backward compatibility
python_to_brace = python_to_condition_b


def convert_condition_a(generated_text: str) -> ConversionResult:
    """
    Condition A uses indentation only - no conversion needed, just extraction.
    This is a placeholder for the extraction logic.
    """
    # For Condition A, extraction cuts at first non-empty line with zero indentation
    # or at end-of-sequence. This is handled at extraction level, not conversion.
    return ConversionResult(
        python_code=generated_text,
        errors=[],
        depth_at_end=0,
        colon_brace_flag=False
    )


def main():
    """Test the converter with sample inputs."""
    # Test cases from §6 and §8 negative controls
    test_cases = [
        # Valid B/C conversion
        ("def process_data(data) {\n    total = 0\n    if len(data) > 0 {\n        total += 1\n    }\n    return total\n}", "valid"),
        
        # Colon before brace (E_COLON_BRACE)
        ("def process_data(data) {\n    total = 0\n    if len(data) > 0: {\n        total += 1\n    }\n    return total\n}", "colon_brace"),
        
        # Python-style header in B/C (E_HEADER_SYNTAX)
        ("def process_data(data) {\n    total = 0\n    if len(data) > 0:\n        total += 1\n    return total\n}", "python_header"),
        
        # Unclosed block (E_UNCLOSED)
        ("def process_data(data) {\n    total = 0\n    if len(data) > 0 {\n        total += 1\n    return total\n}", "unclosed"),
        
        # Braces in statement (should cause syntax error)
        ("def process_data(data) {\n    total = 0\n    if len(data) > 0 {\n        total += {1}\n    }\n    return total\n}", "braces_in_stmt"),
    ]
    
    for i, (text, name) in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {name} ---")
        print("Input:")
        print(text)
        result = convert_brace_to_python(text)
        print("\nOutput:")
        print(result.python_code)
        print(f"\nErrors: {[e.value for e in result.errors]}")
        print(f"Depth at end: {result.depth_at_end}")
        print(f"Colon-brace flag: {result.colon_brace_flag}")


if __name__ == "__main__":
    main()