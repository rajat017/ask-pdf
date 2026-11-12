# HTML Processing Pipeline

## Overview

`htmlToStructuredText(r io.Reader) ([]paragraph, error)` is the main entry point for converting raw HTML into structured `paragraph` slices ready for chunking and embedding. It replaces the previous two-step approach (`htmlToText` + `parseWebIntoParagraphs`) with a single pass that preserves semantic structure from the HTML DOM.

## Processing Stages

### 1. Parse & Find Content Root

```
Raw HTML
  --> html.Parse()
  --> findMainContent()
  --> content root node
```

`findMainContent` walks the full DOM tree once and picks the best content root using this priority:

| Priority | Element | Rationale |
|----------|---------|-----------|
| 1 | `<main>` | Semantic HTML: explicitly marks primary content |
| 2 | Largest `<article>` | Blog posts, news sites often wrap content in `<article>` |
| 3 | `<body>` | Fallback when no semantic containers exist |
| 4 | Document root | Edge case: malformed HTML without `<body>` |

"Largest" is determined by `nodeTextLen`, which sums the trimmed text length of all descendant text nodes.

### 2. Boilerplate Filtering

Two layers of filtering run during the tree walk:

**Tag-based skip** (`skipTags` map):
`script`, `style`, `nav`, `footer`, `header`, `aside`, `form`, `noscript`

When the walker encounters any of these elements, it returns immediately, skipping the entire subtree.

**Class/ID-based skip** (`isBoilerplate`):
Checks `class` and `id` attributes (case-insensitive) for these substrings:
`sidebar`, `menu`, `cookie`, `banner`, `popup`, `modal`, `ad-`, `advertisement`, `nav`, `footer`, `header`

This catches elements like `<div class="cookie-banner">` or `<section id="sidebar-nav">` that use generic tags but serve non-content purposes.

### 3. Tree Walk & Element Dispatch

The main `walk` function recurses through the content root. For each element node, it dispatches to a specialized handler based on tag type:

```
Element Node
  |
  +--> skipTags / isBoilerplate?  --> return (skip subtree)
  |
  +--> h1-h6?                    --> headingHandler
  |
  +--> table?                    --> renderTable
  |
  +--> ul / ol?                  --> renderList
  |
  +--> block element?            --> renderInlineContent
  |
  +--> other                     --> recurse into children
```

The "recurse into children" fallback handles wrapper elements like `<span>`, `<div>` that don't match any category -- it just walks deeper without emitting a paragraph, letting nested block/heading/table/list elements be caught at the right level.

### 4. Heading Handling & Breadcrumb Tracking

When an `<h1>`-`<h6>` tag is encountered:

1. `extractPlainText` collects all text content (ignoring nested markup)
2. `headingStack.push(level, text)` updates the breadcrumb:
   - Pops all entries at the same or deeper level
   - Pushes the new heading

Example breadcrumb evolution:

```
<h1>Products</h1>       --> stack: [Products]            breadcrumb: "Products"
<h2>Enterprise</h2>     --> stack: [Products, Enterprise] breadcrumb: "Products > Enterprise"
<h3>Pricing</h3>        --> stack: [Products, Enterprise, Pricing]
                             breadcrumb: "Products > Enterprise > Pricing"
<h2>Startup</h2>        --> pops Pricing and Enterprise, then pushes Startup
                             stack: [Products, Startup]
                             breadcrumb: "Products > Startup"
```

The heading is emitted as a `paragraph{heading: true, breadcrumb: "..."}`. All subsequent non-heading paragraphs inherit the current breadcrumb until the next heading changes it.

### 5. Table Rendering

`renderTable` converts HTML tables to pipe-delimited markdown format:

```html
<table>
  <tr><th>Name</th><th>Price</th></tr>
  <tr><td>Widget</td><td>$10</td></tr>
</table>
```

Becomes:

```
| Name | Price |
| --- | --- |
| Widget | $10 |
```

Processing steps:
1. Walk the table subtree collecting `<tr>` rows, each containing `<td>`/`<th>` cells
2. Normalize all rows to the same column count (pad with empty strings)
3. First row is treated as header, followed by a `| --- | --- |` separator
4. Cell content is rendered via `renderInlineContent` (preserving links etc.)

**Large table splitting** (`splitTableIfLarge`): If the rendered table exceeds 1200 characters, it's split into multiple paragraphs. Each part repeats the header + separator lines so context isn't lost during chunking.

### 6. List Rendering

`renderList` handles `<ul>` and `<ol>` with recursive nesting:

```html
<ul>
  <li>First item
    <ul>
      <li>Nested item</li>
    </ul>
  </li>
  <li>Second item</li>
</ul>
```

Becomes:

```
- First item
  - Nested item
- Second item
```

- Unordered lists use `- ` prefix
- Ordered lists use `1. `, `2. `, etc.
- Each nesting level adds 2 spaces of indentation
- Inline content and nested lists within an `<li>` are processed separately

### 7. Inline Content & Link Preservation

`renderInlineContent` handles the text within block elements. It:

- Collapses whitespace (multiple spaces/newlines become a single space)
- Preserves links: `<a href="url">text</a>` becomes `text (url)`
- Skips anchor-only links (`href="#..."`) -- just keeps the text
- Converts `<br>` to newline
- Stops at nested block elements, headings, tables, lists (those are handled by the main walker)

**URL shortening** (`shortenURL`): If a link URL exceeds 100 characters, query parameters and fragment are stripped first. If still too long, the URL is truncated with `...`.

## Output

The function returns `[]paragraph` where each entry has:

| Field | Description |
|-------|-------------|
| `text` | The rendered text content |
| `heading` | `true` for h1-h6 headings |
| `breadcrumb` | Hierarchical heading path, e.g. `"Products > Enterprise > Pricing"` |
| `pageNum` | Always `0` (web pages have no page numbers) |

These paragraphs feed directly into `buildChunks()`, which groups them into embedding-sized chunks with overlap, section tracking, and parent-text expansion.

## Helper Function Reference

| Function | Purpose |
|----------|---------|
| `findMainContent` | Locate content root (`<main>` > largest `<article>` > `<body>`) |
| `nodeTextLen` | Count text characters under a node |
| `isBoilerplate` | Check class/id for boilerplate patterns |
| `headingLevel` | Map tag name to heading level (1-6 or 0) |
| `extractPlainText` | Get raw text from a node, ignoring all markup |
| `renderInlineContent` | Render inline content with link preservation |
| `renderTable` | Convert `<table>` to pipe-format markdown |
| `splitTableIfLarge` | Split large tables into chunks with repeated headers |
| `renderList` | Convert `<ul>`/`<ol>` to indented text with bullets/numbers |
| `shortenURL` | Trim long URLs by removing query/fragment |
| `collapseWhitespace` | Replace whitespace runs with single space |
| `normalizeSpaces` | Trim + collapse spaces (preserves newlines) |
