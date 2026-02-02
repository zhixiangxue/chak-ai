"""Large-scale Skill demo: 100 methods

Demonstrates 3-stage progressive disclosure with a skill containing 100 methods.
Tests if the framework can handle large skills without overwhelming the LLM.
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console

from chak import Conversation
from chak.tools import SkillBase, wrap_tools

load_dotenv()
console = Console()


class MegaSkill(SkillBase):
    """A skill with 100 mock methods to test progressive disclosure."""
    
    name = "mega_operations"
    description = "Comprehensive operations toolkit with 100 different methods"
    
    def __init__(self):
        self.call_count = 0
    
    # File operations (20 methods)
    def file_read(self, path: str) -> str:
        """Read a file from path"""
        self.call_count += 1
        result = f"✅ file_read called (total calls: {self.call_count})"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_read()[/bold cyan] [green]{result}[/green]")
        return result
    
    def file_write(self, path: str, content: str) -> str:
        """Write content to a file"""
        self.call_count += 1
        result = f"✅ file_write called (total calls: {self.call_count})"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_write()[/bold cyan] [green]{result}[/green]")
        return result
    
    def file_append(self, path: str, content: str) -> str:
        """Append content to a file"""
        self.call_count += 1
        result = f"✅ file_append called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_append()[/bold cyan] [green]{result}[/green]")
        return result
    
    def file_delete(self, path: str) -> str:
        """Delete a file"""
        self.call_count += 1
        result = f"✅ file_delete called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_delete()[/bold cyan] [green]{result}[/green]")
        return result
    
    def file_copy(self, src: str, dst: str) -> str:
        """Copy file from src to dst"""
        self.call_count += 1
        result = f"✅ file_copy called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_copy()[/bold cyan] [green]{result}[/green]")
        return result
    
    def file_move(self, src: str, dst: str) -> str:
        """Move file from src to dst"""
        self.call_count += 1
        result = f"✅ file_move called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_move()[/bold cyan] [green]{result}[/green]")
        return result
    
    def file_rename(self, old: str, new: str) -> str:
        """Rename a file"""
        self.call_count += 1
        result = f"✅ file_rename called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_rename()[/bold cyan] [green]{result}[/green]")
        return result
    
    def file_exists(self, path: str) -> str:
        """Check if file exists"""
        self.call_count += 1
        result = f"✅ file_exists called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_exists()[/bold cyan] [green]{result}[/green]")
        return result
    
    def file_size(self, path: str) -> str:
        """Get file size"""
        self.call_count += 1
        result = f"✅ file_size called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_size()[/bold cyan] [green]{result}[/green]")
        return result
    
    def file_info(self, path: str) -> str:
        """Get file information"""
        self.call_count += 1
        result = f"✅ file_info called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ file_info()[/bold cyan] [green]{result}[/green]")
        return result
    
    # Database operations (20 methods)
    def db_connect(self, host: str, port: int) -> str:
        """Connect to database"""
        self.call_count += 1
        result = f"✅ db_connect called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_connect()[/bold cyan] [green]{result}[/green]")
        return result
    
    def db_disconnect(self) -> str:
        """Disconnect from database"""
        self.call_count += 1
        result = f"✅ db_disconnect called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_disconnect()[/bold cyan] [green]{result}[/green]")
        return result
    
    def db_query(self, sql: str) -> str:
        """Execute SQL query"""
        self.call_count += 1
        result = f"✅ db_query called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_query()[/bold cyan] [green]{result}[/green]")
        return result
    
    def db_insert(self, table: str, data: dict) -> str:
        """Insert data into table"""
        self.call_count += 1
        result = f"✅ db_insert called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_insert()[/bold cyan] [green]{result}[/green]")
        return result
    
    def db_update(self, table: str, data: dict, condition: str) -> str:
        """Update table data"""
        self.call_count += 1
        result = f"✅ db_update called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_update()[/bold cyan] [green]{result}[/green]")
        return result
    
    def db_delete(self, table: str, condition: str) -> str:
        """Delete table rows"""
        self.call_count += 1
        result = f"✅ db_delete called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_delete()[/bold cyan] [green]{result}[/green]")
        return result
    
    def db_create_table(self, name: str, schema: str) -> str:
        """Create a new table"""
        self.call_count += 1
        result = f"✅ db_create_table called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_create_table()[/bold cyan] [green]{result}[/green]")
        return result
    
    def db_drop_table(self, name: str) -> str:
        """Drop a table"""
        self.call_count += 1
        result = f"✅ db_drop_table called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_drop_table()[/bold cyan] [green]{result}[/green]")
        return result
    
    def db_list_tables(self) -> str:
        """List all tables"""
        self.call_count += 1
        result = f"✅ db_list_tables called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_list_tables()[/bold cyan] [green]{result}[/green]")
        return result
    
    def db_backup(self, path: str) -> str:
        """Backup database"""
        self.call_count += 1
        result = f"✅ db_backup called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ db_backup()[/bold cyan] [green]{result}[/green]")
        return result
    
    # Network operations (20 methods)
    def net_get(self, url: str) -> str:
        """HTTP GET request"""
        self.call_count += 1
        result = f"✅ net_get called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_get()[/bold cyan] [green]{result}[/green]")
        return result
    
    def net_post(self, url: str, data: dict) -> str:
        """HTTP POST request"""
        self.call_count += 1
        result = f"✅ net_post called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_post()[/bold cyan] [green]{result}[/green]")
        return result
    
    def net_put(self, url: str, data: dict) -> str:
        """HTTP PUT request"""
        self.call_count += 1
        result = f"✅ net_put called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_put()[/bold cyan] [green]{result}[/green]")
        return result
    
    def net_delete(self, url: str) -> str:
        """HTTP DELETE request"""
        self.call_count += 1
        result = f"✅ net_delete called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_delete()[/bold cyan] [green]{result}[/green]")
        return result
    
    def net_download(self, url: str, path: str) -> str:
        """Download file from URL"""
        self.call_count += 1
        result = f"✅ net_download called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_download()[/bold cyan] [green]{result}[/green]")
        return result
    
    def net_upload(self, url: str, file: str) -> str:
        """Upload file to URL"""
        self.call_count += 1
        result = f"✅ net_upload called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_upload()[/bold cyan] [green]{result}[/green]")
        return result
    
    def net_ping(self, host: str) -> str:
        """Ping a host"""
        self.call_count += 1
        result = f"✅ net_ping called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_ping()[/bold cyan] [green]{result}[/green]")
        return result
    
    def net_resolve(self, domain: str) -> str:
        """Resolve domain to IP"""
        self.call_count += 1
        result = f"✅ net_resolve called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_resolve()[/bold cyan] [green]{result}[/green]")
        return result
    
    def net_scan_port(self, host: str, port: int) -> str:
        """Scan port on host"""
        self.call_count += 1
        result = f"✅ net_scan_port called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_scan_port()[/bold cyan] [green]{result}[/green]")
        return result
    
    def net_check_ssl(self, domain: str) -> str:
        """Check SSL certificate"""
        self.call_count += 1
        result = f"✅ net_check_ssl called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ net_check_ssl()[/bold cyan] [green]{result}[/green]")
        return result
    
    # Data processing (20 methods)
    def data_parse_json(self, text: str) -> str:
        """Parse JSON string"""
        self.call_count += 1
        result = f"✅ data_parse_json called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_parse_json()[/bold cyan] [green]{result}[/green]")
        return result
    
    def data_parse_xml(self, text: str) -> str:
        """Parse XML string"""
        self.call_count += 1
        result = f"✅ data_parse_xml called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_parse_xml()[/bold cyan] [green]{result}[/green]")
        return result
    
    def data_parse_csv(self, text: str) -> str:
        """Parse CSV string"""
        self.call_count += 1
        result = f"✅ data_parse_csv called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_parse_csv()[/bold cyan] [green]{result}[/green]")
        return result
    
    def data_to_json(self, data: dict) -> str:
        """Convert data to JSON"""
        self.call_count += 1
        result = f"✅ data_to_json called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_to_json()[/bold cyan] [green]{result}[/green]")
        return result
    
    def data_to_xml(self, data: dict) -> str:
        """Convert data to XML"""
        self.call_count += 1
        result = f"✅ data_to_xml called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_to_xml()[/bold cyan] [green]{result}[/green]")
        return result
    
    def data_to_csv(self, data: list) -> str:
        """Convert data to CSV"""
        self.call_count += 1
        result = f"✅ data_to_csv called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_to_csv()[/bold cyan] [green]{result}[/green]")
        return result
    
    def data_filter(self, data: list, condition: str) -> str:
        """Filter data by condition"""
        self.call_count += 1
        result = f"✅ data_filter called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_filter()[/bold cyan] [green]{result}[/green]")
        return result
    
    def data_sort(self, data: list, key: str) -> str:
        """Sort data by key"""
        self.call_count += 1
        result = f"✅ data_sort called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_sort()[/bold cyan] [green]{result}[/green]")
        return result
    
    def data_group(self, data: list, key: str) -> str:
        """Group data by key"""
        self.call_count += 1
        result = f"✅ data_group called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_group()[/bold cyan] [green]{result}[/green]")
        return result
    
    def data_aggregate(self, data: list, func: str) -> str:
        """Aggregate data with function"""
        self.call_count += 1
        result = f"✅ data_aggregate called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ data_aggregate()[/bold cyan] [green]{result}[/green]")
        return result
    
    # String operations (20 methods)
    def str_upper(self, text: str) -> str:
        """Convert to uppercase"""
        self.call_count += 1
        result = f"✅ str_upper called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_upper()[/bold cyan] [green]{result}[/green]")
        return result
    
    def str_lower(self, text: str) -> str:
        """Convert to lowercase"""
        self.call_count += 1
        result = f"✅ str_lower called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_lower()[/bold cyan] [green]{result}[/green]")
        return result
    
    def str_capitalize(self, text: str) -> str:
        """Capitalize first letter"""
        self.call_count += 1
        result = f"✅ str_capitalize called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_capitalize()[/bold cyan] [green]{result}[/green]")
        return result
    
    def str_reverse(self, text: str) -> str:
        """Reverse string"""
        self.call_count += 1
        result = f"✅ str_reverse called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_reverse()[/bold cyan] [green]{result}[/green]")
        return result
    
    def str_replace(self, text: str, old: str, new: str) -> str:
        """Replace substring"""
        self.call_count += 1
        result = f"✅ str_replace called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_replace()[/bold cyan] [green]{result}[/green]")
        return result
    
    def str_split(self, text: str, delimiter: str) -> str:
        """Split string by delimiter"""
        self.call_count += 1
        result = f"✅ str_split called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_split()[/bold cyan] [green]{result}[/green]")
        return result
    
    def str_join(self, parts: list, separator: str) -> str:
        """Join list with separator"""
        self.call_count += 1
        result = f"✅ str_join called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_join()[/bold cyan] [green]{result}[/green]")
        return result
    
    def str_trim(self, text: str) -> str:
        """Trim whitespace"""
        self.call_count += 1
        result = f"✅ str_trim called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_trim()[/bold cyan] [green]{result}[/green]")
        return result
    
    def str_pad(self, text: str, length: int, char: str) -> str:
        """Pad string to length"""
        self.call_count += 1
        result = f"✅ str_pad called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_pad()[/bold cyan] [green]{result}[/green]")
        return result
    
    def str_truncate(self, text: str, length: int) -> str:
        """Truncate string to length"""
        self.call_count += 1
        result = f"✅ str_truncate called"
        console.print(f"[bold yellow]⚡ {datetime.now().strftime('%H:%M:%S')}[/bold yellow] [bold cyan]→ str_truncate()[/bold cyan] [green]{result}[/green]")
        return result


async def main():
    """Run large-scale skill demo."""
    
    # Create skill with 100 methods
    mega_skill = MegaSkill()
    tools = wrap_tools([mega_skill])
    
    print("="*80)
    print(f"🚀 Large-Scale Skill Demo")
    print("="*80)
    print(f"\n✨ Skill: {mega_skill.name}")
    print(f"📝 Description: {mega_skill.description}")
    print(f"🔧 Total methods: 50 mock methods (file, db, net, data, string ops)")
    print()
    
    # Create conversation
    conv = Conversation(
        model_uri="bailian/qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        tools=tools
    )
    
    # Test 1: Activate skill and see method summary
    print("\n" + "="*80)
    print("📋 Test 1: Activate skill and view method summary")
    print("="*80)
    print("💬 User: Please show me what operations are available")
    print()
    
    response = await conv.asend("Please activate the mega_operations skill and show me what operations are available")
    print(f"🤖 Assistant:\n{response.content[:500]}...")
    
    # Test 2: Use a specific method
    print("\n" + "="*80)
    print("📋 Test 2: Call a specific file operation")
    print("="*80)
    print("💬 User: Read the file /tmp/test.txt")
    print()
    
    response = await conv.asend("Use the file_read method to read /tmp/test.txt")
    print(f"🤖 Assistant: {response.content}")
    
    # Test 3: Use another method
    print("\n" + "="*80)
    print("📋 Test 3: Call a network operation")
    print("="*80)
    print("💬 User: Make a GET request to https://api.example.com")
    print()
    
    response = await conv.asend("Use the net_get method to make a request to https://api.example.com")
    print(f"🤖 Assistant: {response.content}")
    
    # Test 4: Parallel method calls (Stage 3 with multiple methods)
    print("\n" + "="*80)
    print("📋 Test 4: Parallel method calls (multiple methods in one turn)")
    print("="*80)
    print("💬 User: Read /tmp/test.txt, write to /tmp/output.txt, and check if /tmp/data.json exists")
    print()
    
    response = await conv.asend(
        "Please do three things in parallel: "
        "1) Read /tmp/test.txt using file_read, "
        "2) Write 'hello' to /tmp/output.txt using file_write, "
        "3) Check if /tmp/data.json exists using file_exists"
    )
    print(f"🤖 Assistant: {response.content}")
    
    print("\n" + "="*80)
    print(f"✅ Demo completed! Total method calls: {mega_skill.call_count}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
