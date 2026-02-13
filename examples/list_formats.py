# Example: List Available Formats

"""
This example demonstrates how to discover available
quantization formats for each backend.
"""

from qwodel import Quantizer
from rich.console import Console
from rich.table import Table


def main():
    console = Console()
    
    console.print("\n[bold cyan]📋 Available Quantization Formats[/bold cyan]\n")
    
    # Get all formats
    all_formats = Quantizer.list_formats()
    
    for backend, formats in all_formats.items():
        console.print(f"\n[bold yellow]{backend.upper()} Backend:[/bold yellow]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Format", style="cyan", width=15)
        table.add_column("Description", style="white")
        
        for fmt_name, description in formats.items():
            table.add_row(fmt_name, description)
        
        console.print(table)
    
    # List backends
    console.print("\n[bold cyan]🔧 Available Backends:[/bold cyan]")
    backends = Quantizer.list_backends()
    for backend in backends:
        console.print(f"  • {backend}")
    
    console.print()


if __name__ == "__main__":
    main()
