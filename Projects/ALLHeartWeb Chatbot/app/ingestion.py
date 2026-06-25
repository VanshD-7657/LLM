import os
import re
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Any
from app.config import DOCS_DIR, KNOWLEDGE_DIR, CHUNK_SIZE_MIN, CHUNK_SIZE_MAX, CHUNK_OVERLAP

def clean_hyphenated_content(content: str) -> str:
    """Detect and clean up text that has single characters separated by hyphens (e.g. -A-l-l- -e-n-d-p-o-i-n-t-s-)."""
    cleaned_lines = []
    for line in content.splitlines():
        # Check ratio of dashes to alpha characters
        dash_count = line.count('-')
        alpha_count = sum(1 for c in line if c.isalpha())
        
        # If the line has significant dashes and they are mostly single characters separated by dashes
        if dash_count > 5 and alpha_count > 5 and (dash_count / alpha_count) > 0.4:
            # Replace "- -" with " " (space)
            cleaned = line.replace('- -', ' ')
            # Remove all remaining hyphens
            cleaned = cleaned.replace('-', '')
            line = cleaned
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def parse_markdown_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown files."""
    # Matches patterns like:
    # ---
    # title: WHOIS Lookup
    # ---
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
    if match:
        yaml_content = match.group(1)
        body = match.group(2)
        try:
            metadata = yaml.safe_load(yaml_content)
            if isinstance(metadata, dict):
                return metadata, body
        except Exception:
            pass
    return {}, content

def parse_knowledge_text(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse source URL and metadata from knowledge text files."""
    metadata = {}
    lines = content.split('\n')
    body_lines = []
    
    parsing_headers = True
    for line in lines:
        if parsing_headers:
            stripped = line.strip()
            if not stripped:
                # Empty line ends the header section if we've already found some metadata
                if metadata:
                    parsing_headers = False
                continue
            
            # Look for headers like "Source URL: ..."
            if ":" in stripped:
                key, val = stripped.split(":", 1)
                key = key.strip().lower()
                val = val.strip()
                if key == "source url":
                    metadata["source_url"] = val
                elif key == "page type":
                    metadata["page_type"] = val
                elif key == "scraped at":
                    metadata["scraped_at"] = val
                else:
                    # Not a header we recognize, treat as body start
                    parsing_headers = False
                    body_lines.append(line)
            else:
                parsing_headers = False
                body_lines.append(line)
        else:
            body_lines.append(line)
            
    body = '\n'.join(body_lines)
    return metadata, body

def split_text_into_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Split document text into structural semantic blocks:
    - Headers (e.g. # Header, ## Subheader)
    - Code blocks (```bash ... ```)
    - Tables (| Column 1 | Column 2 |)
    - Paragraphs/lists (non-blank text lines)
    
    Returns a list of (block_text, block_type) tuples.
    """
    lines = text.split('\n')
    blocks = []
    current_block_lines = []
    in_code_block = False
    in_table = False
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Code Block boundary check
        if stripped.startswith('```'):
            if in_code_block:
                current_block_lines.append(line)
                blocks.append(('\n'.join(current_block_lines), "code"))
                current_block_lines = []
                in_code_block = False
            else:
                if current_block_lines:
                    blocks.append(('\n'.join(current_block_lines), "table" if in_table else "text"))
                    current_block_lines = []
                    in_table = False
                current_block_lines.append(line)
                in_code_block = True
            continue
            
        if in_code_block:
            current_block_lines.append(line)
            continue
            
        # 2. Table boundary check
        if stripped.startswith('|'):
            if not in_table:
                if current_block_lines:
                    blocks.append(('\n'.join(current_block_lines), "text"))
                    current_block_lines = []
                in_table = True
            current_block_lines.append(line)
            continue
        else:
            if in_table:
                blocks.append(('\n'.join(current_block_lines), "table"))
                current_block_lines = []
                in_table = False
                
        # 3. Header check
        if stripped.startswith('#'):
            if current_block_lines:
                blocks.append(('\n'.join(current_block_lines), "text"))
                current_block_lines = []
            blocks.append((line, "header"))
            continue
            
        # 4. Empty line (separates paragraphs)
        if not stripped:
            if current_block_lines:
                blocks.append(('\n'.join(current_block_lines), "text"))
                current_block_lines = []
        else:
            current_block_lines.append(line)
            
    # Cleanup trailing blocks
    if current_block_lines:
        blocks.append(('\n'.join(current_block_lines), "code" if in_code_block else ("table" if in_table else "text")))
        
    return blocks

def chunk_document_blocks(blocks: List[Tuple[str, str]], base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Groups semantic blocks into chunks of size 800 - 1200 characters.
    Ensures code blocks and tables are never split.
    Uses sliding window for 150 - 200 characters overlap.
    """
    chunks = []
    current_chunk_blocks = []
    current_chunk_len = 0
    current_section = base_metadata.get("title", "")
    
    i = 0
    while i < len(blocks):
        block_text, block_type = blocks[i]
        
        # Track the current active section header
        if block_type == "header":
            # Clean up header markers (e.g. ## API Lookup -> API Lookup)
            current_section = re.sub(r'^#+\s*', '', block_text).strip()
            
        # If adding this block exceeds max limit and we already have a reasonable chunk,
        # we need to emit the current chunk.
        if current_chunk_len + len(block_text) > CHUNK_SIZE_MAX and current_chunk_len >= CHUNK_SIZE_MIN:
            # Emit current chunk
            chunk_content = "\n\n".join([b[0] for b in current_chunk_blocks])
            
            chunk_metadata = base_metadata.copy()
            chunk_metadata["section"] = current_section
            
            chunks.append({
                "content": chunk_content,
                "metadata": chunk_metadata
            })
            
            # Implement overlap: backtrack to include recent blocks that sum up to ~150-200 chars.
            overlap_len = 0
            overlap_blocks = []
            for ob in reversed(current_chunk_blocks):
                # Don't overlap massive blocks or header blocks that we can just regenerate
                if len(ob[0]) > 400 or ob[1] == "header":
                    continue
                if overlap_len + len(ob[0]) <= CHUNK_OVERLAP:
                    overlap_blocks.insert(0, ob)
                    overlap_len += len(ob[0])
                else:
                    break
            
            current_chunk_blocks = overlap_blocks
            current_chunk_len = overlap_len
            
        # Add the block
        current_chunk_blocks.append((block_text, block_type))
        current_chunk_len += len(block_text) + 2  # +2 accounts for join newlines
        i += 1
        
    # Emit final chunk if it contains anything
    if current_chunk_blocks:
        chunk_content = "\n\n".join([b[0] for b in current_chunk_blocks])
        chunk_metadata = base_metadata.copy()
        chunk_metadata["section"] = current_section
        
        chunks.append({
            "content": chunk_content,
            "metadata": chunk_metadata
        })
        
    return chunks

def ingest_docs(docs_dir: Path) -> List[Dict[str, Any]]:
    """Loads and chunks all markdown documents under docs/ directory, excluding combined_reference.md."""
    all_chunks = []
    
    # Recursive search for .md files
    for filepath in docs_dir.rglob("*.md"):
        # Exclude combined reference file to prevent duplication
        if filepath.name == "combined_reference.md":
            continue
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            content = clean_hyphenated_content(content)
            frontmatter, body = parse_markdown_frontmatter(content)
            
            # Determine category from frontmatter, directory or path
            category = frontmatter.get("category", "")
            if not category:
                # Use parent directory name as category
                parent_dir = filepath.parent.name
                if parent_dir != "docs" and parent_dir != "":
                    category = parent_dir.replace("_", " ").title()
                else:
                    category = "General"
                    
            source_url = frontmatter.get("source_url", "")
            if not source_url:
                # Generate relative path as fallback
                rel_path = filepath.relative_to(docs_dir.parent)
                source_url = rel_path.as_posix()
                
            base_metadata = {
                "source": source_url,
                "category": category,
                "file_name": filepath.name,
                "document_type": "api_docs",
                "title": frontmatter.get("title", filepath.stem.replace("_", " ").title())
            }
            
            blocks = split_text_into_blocks(body)
            chunks = chunk_document_blocks(blocks, base_metadata)
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Error reading doc file {filepath}: {e}")
            
    return all_chunks

def ingest_knowledge(knowledge_dir: Path) -> List[Dict[str, Any]]:
    """Loads and chunks all text documents under knowledge/ directory."""
    all_chunks = []
    
    for filepath in knowledge_dir.glob("*.txt"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            knowledge_headers, body = parse_knowledge_text(content)
            
            source_url = knowledge_headers.get("source_url", "")
            if not source_url:
                rel_path = filepath.relative_to(knowledge_dir.parent)
                source_url = rel_path.as_posix()
                
            category = knowledge_headers.get("page_type", "General Company Knowledge")
            
            base_metadata = {
                "source": source_url,
                "category": category,
                "file_name": filepath.name,
                "document_type": "company_knowledge",
                "title": filepath.stem.replace("_", " ").title()
            }
            
            blocks = split_text_into_blocks(body)
            chunks = chunk_document_blocks(blocks, base_metadata)
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Error reading knowledge file {filepath}: {e}")
            
    return all_chunks

def ingest_all() -> List[Dict[str, Any]]:
    """Ingests both docs and knowledge directories and returns all chunks."""
    print("Ingesting API documentation...")
    doc_chunks = ingest_docs(DOCS_DIR)
    print(f"Ingested {len(doc_chunks)} chunks from API documentation.")
    
    print("Ingesting company knowledge...")
    knowledge_chunks = ingest_knowledge(KNOWLEDGE_DIR)
    print(f"Ingested {len(knowledge_chunks)} chunks from company knowledge.")
    
    return doc_chunks + knowledge_chunks
