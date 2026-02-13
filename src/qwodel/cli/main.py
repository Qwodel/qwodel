"""
Qwodel CLI

Command-line interface for model quantization.
"""

import click
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from qwodel import Quantizer, __version__
from qwodel.backends import BackendRegistry
from qwodel.core.exceptions import QuantizationError

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="qwodel")
def cli():
    """
    Qwodel - Production-Grade Model Quantization
    
    Quantize models for AWQ (GPU), GGUF (CPU), and CoreML (Apple devices).
    """
    pass


@cli.command()
@click.option(
    "--backend", "-b",
    required=True,
    type=click.Choice(["gguf", "awq", "coreml"], case_sensitive=False),
    help="Quantization backend to use"
)
@click.option(
    "--format", "-f",
    required=True,
    help="Quantization format (e.g., Q4_K_M, int4, float16)"
)
@click.option(
    "--model", "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to source model (directory or file)"
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output directory for quantized model"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
def quantize(
    backend: str,
    format: str,
    model: str,
    output: str,
    verbose: bool
):
    """
    Quantize a model with specified backend and format.
    
    Examples:
    
        # GGUF quantization (CPU)
        qwodel quantize -b gguf -f Q4_K_M -m ./model -o ./output
        
        # AWQ quantization (GPU)
        qwodel quantize -b awq -f int4 -m ./model -o ./output
        
        # CoreML quantization (Apple)
        qwodel quantize -b coreml -f float16 -m ./model -o ./output
    """
    try:
        console.print(f"[bold cyan]Qwodel v{__version__}[/bold cyan]")
        console.print(f"Backend: [yellow]{backend}[/yellow]")
        console.print(f"Format: [yellow]{format}[/yellow]")
        console.print(f"Model: [blue]{model}[/blue]")
        console.print(f"Output: [blue]{output}[/blue]\n")
        
        # Progress tracking
        current_progress = {"percent": 0, "stage": "", "message": ""}
        
        def progress_callback(percent: int, stage: str, message: str = ""):
            current_progress["percent"] = percent
            current_progress["stage"] = stage
            current_progress["message"] = message
            if verbose:
                console.print(f"[{percent}%] {stage}: {message}")
        
        # Create quantizer
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(
                "[cyan]Quantizing model...",
                total=100
            )
            
            # Update progress based on callback
            def update_progress(percent: int, stage: str, message: str = ""):
                progress.update(
                    task,
                    completed=percent,
                    description=f"[cyan]{stage.replace('_', ' ').title()}"
                )
                if verbose and message:
                    console.print(f"  {message}")
            
            quantizer = Quantizer(
                backend=backend,
                model_path=model,
                output_dir=output,
                progress_callback=update_progress
            )
            
            output_path = quantizer.quantize(format=format)
        
        # Success message
        console.print("\n[bold green]✅ Quantization successful![/bold green]")
        console.print(f"Output: [blue]{output_path}[/blue]")
        
        # Get model info
        info = quantizer.get_model_info()
        if info.get("file_size"):
            size_mb = info["file_size"] / (1024 * 1024)
            console.print(f"Size: [yellow]{size_mb:.2f} MB[/yellow]")
        
    except QuantizationError as e:
        console.print(f"\n[bold red]❌ Quantization failed:[/bold red] {e}", err=True)
        raise click.Abort()
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}", err=True)
        if verbose:
            console.print_exception()
        raise click.Abort()


@cli.command("list-formats")
@click.option(
    "--backend", "-b",
    type=click.Choice(["gguf", "awq", "coreml"], case_sensitive=False),
    help="Filter by specific backend (optional)"
)
def list_formats(backend: Optional[str]):
    """
    List available quantization formats.
    
    Examples:
    
        # List all formats
        qwodel list-formats
        
        # List GGUF formats only
        qwodel list-formats --backend gguf
    """
    formats = Quantizer.list_formats(backend=backend)
    
    for backend_name, fmt_dict in formats.items():
        console.print(f"\n[bold cyan]{backend_name.upper()} Formats:[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Format", style="yellow")
        table.add_column("Description", style="white")
        
        for fmt_name, description in fmt_dict.items():
            table.add_row(fmt_name, description)
        
        console.print(table)


@cli.command("list-backends")
def list_backends():
    """
    List available quantization backends.
    
    Example:
    
        qwodel list-backends
    """
    backends = BackendRegistry.list_backends()
    
    console.print("\n[bold cyan]Available Backends:[/bold cyan]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Backend", style="yellow")
    table.add_column("Status", style="green")
    
    for backend_name in backends:
        table.add_row(backend_name, "✓ Available")
    
    console.print(table)
    console.print(f"\nTotal: [yellow]{len(backends)}[/yellow] backends\n")


if __name__ == "__main__":
    cli()
