import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table

from conversation import DEFAULT_MODEL, DocuMindEngine, ollama_status
from extract import find_pdfs
from library import DocumentLibrary

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

console = Console()

HELP = """[bold]Commands[/bold]
  /add <path>     index more PDFs or a folder
  /docs           list indexed documents
  /summary <file> summarize a document
  /suggest        get starter questions
  /find <text>    search chunks without the LLM
  /save <file>    export the conversation as markdown
  /clear          forget the conversation
  /help           show this list
  /exit           quit"""


def banner() -> None:
    console.print(Panel.fit(
        "[bold cyan]DocuMind[/bold cyan]  [dim]offline PDF answers with page citations[/dim]",
        border_style="cyan",
    ))


def show_documents(library: DocumentLibrary) -> None:
    documents = library.documents()
    if not documents:
        console.print("[yellow]No documents indexed.[/yellow]")
        return

    table = Table(border_style="dim", header_style="bold cyan")
    table.add_column("Document")
    table.add_column("Pages", justify="right")
    table.add_column("Words", justify="right")
    for doc in documents:
        table.add_row(doc["filename"], str(doc["total_pages"]), f"{doc['words']:,}")
    console.print(table)

    stats = library.stats()
    console.print(f"[dim]{stats['chunks']} chunks indexed[/dim]")


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    table = Table(border_style="dim", header_style="bold magenta", show_lines=False)
    table.add_column("Source", style="cyan", max_width=26)
    table.add_column("Page", justify="center", width=8)
    table.add_column("Match", justify="right", width=7)
    table.add_column("Excerpt", style="dim")
    for source in sources:
        table.add_row(
            source["source"],
            f"{source['page']}/{source['total_pages']}",
            f"{source['score'] * 100:.0f}%",
            source["snippet"].replace("\n", " "),
        )
    console.print(table)


def index_paths(library: DocumentLibrary, paths: list[str]) -> None:
    stored = []
    for path in paths:
        with open(path, "rb") as handle:
            stored.append(library.store_upload(path, handle))
    with Status("[cyan]Extracting text and building embeddings...[/cyan]", spinner="dots"):
        result = library.add(stored)
    console.print(f"[green]Indexed {result['pages']} pages from {len(stored)} file(s).[/green]")


def handle_command(line: str, library: DocumentLibrary, engine: DocuMindEngine) -> bool:
    command, _, argument = line[1:].partition(" ")
    command = command.lower()
    argument = argument.strip()

    if command in ("exit", "quit", "q"):
        return False

    if command == "help":
        console.print(HELP)

    elif command == "docs":
        show_documents(library)

    elif command == "add":
        paths = find_pdfs(argument) if argument else []
        if not paths:
            console.print("[red]No PDFs found at that path.[/red]")
        else:
            index_paths(library, paths)
            engine.set_index(library.index)

    elif command == "summary":
        pages = library.pages_for(argument)
        if not pages:
            console.print("[red]Document not found. Use /docs to see the list.[/red]")
        else:
            with Status("[cyan]Summarizing...[/cyan]", spinner="dots"):
                summary = engine.summarize(pages)
            console.print(Panel(Markdown(summary), title=argument, border_style="cyan"))

    elif command == "suggest":
        documents = library.documents()
        if not documents:
            console.print("[yellow]Index a document first.[/yellow]")
        else:
            with Status("[cyan]Thinking of questions...[/cyan]", spinner="dots"):
                questions = engine.suggest(library.pages_for(documents[0]["filename"]))
            for question in questions:
                console.print(f"  [cyan]•[/cyan] {question}")

    elif command == "find":
        hits = engine.search(argument, k=6)
        if not hits:
            console.print("[yellow]Nothing matched.[/yellow]")
        else:
            show_sources(hits)

    elif command == "save":
        target = argument or "documind-chat.md"
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(engine.memory.as_markdown())
        console.print(f"[green]Saved conversation to {target}[/green]")

    elif command == "clear":
        engine.clear_history()
        console.print("[yellow]Conversation cleared.[/yellow]")

    else:
        console.print(f"[red]Unknown command: /{command}[/red] — try /help")

    return True


def chat_loop(library: DocumentLibrary, engine: DocuMindEngine, k: int) -> None:
    console.print("\n[green]Ready.[/green] Ask a question, or type /help for commands.\n")

    while True:
        try:
            line = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            return

        if not line:
            continue

        if line.startswith("/"):
            if not handle_command(line, library, engine):
                console.print("[dim]Bye.[/dim]")
                return
            console.print()
            continue

        console.print("\n[bold green]documind[/bold green]")
        sources: list[dict] = []
        elapsed = 0.0
        failed = False

        for event in engine.ask_stream(line, k=k):
            if event["type"] == "error":
                console.print(f"[red]{event['message']}[/red]")
                failed = True
                break
            if event["type"] == "token":
                console.print(event["text"], end="", markup=False, highlight=False)
            elif event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "done":
                elapsed = event["elapsed"]

        if failed:
            console.print()
            continue

        console.print(f"\n[dim]{elapsed}s[/dim]")
        show_sources(sources)
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind - offline PDF assistant")
    parser.add_argument("--web", action="store_true", help="launch the web interface")
    parser.add_argument("--pdf", help="PDF files or a folder to index on startup")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--k", type=int, default=4, help="chunks to retrieve per question")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fresh", action="store_true", help="clear the library before starting")
    args = parser.parse_args()

    if args.web:
        console.print(f"[cyan]DocuMind running at http://{args.host}:{args.port}[/cyan]")
        import uvicorn

        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
        return

    banner()

    status = ollama_status()
    if not status["running"]:
        console.print(Panel(status["error"], title="Ollama offline", border_style="red"))
        sys.exit(1)
    console.print(f"[green]Ollama online[/green] [dim]({', '.join(status['models']) or 'no models pulled'})[/dim]")

    library = DocumentLibrary()
    if args.fresh:
        library.reset()

    if args.pdf:
        paths = find_pdfs(args.pdf)
        if not paths:
            console.print("[red]No PDFs found at that path.[/red]")
            sys.exit(1)
        index_paths(library, paths)

    if not library.documents():
        console.print("\n[dim]No documents yet. Give me a PDF path or folder (blank to skip).[/dim]")
        answer = Prompt.ask("[cyan]path[/cyan]", default="").strip()
        paths = find_pdfs(answer) if answer else []
        if paths:
            index_paths(library, paths)

    show_documents(library)
    engine = DocuMindEngine(index=library.index, model=args.model)
    chat_loop(library, engine, args.k)


if __name__ == "__main__":
    main()
