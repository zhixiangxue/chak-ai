"""
Comprehensive test for document attachments - local and remote files
Tests: PDF, DOC, TXT, Excel, CSV, Link
"""
import asyncio
import os
from pathlib import Path
from chak.attachment import PDF, DOC, Excel, CSV, TXT, Link

# Test file paths
BASE_DIR = Path(__file__).parent.parent
TMP_DIR = BASE_DIR / "tmp"
OSS_BASE_URL = "https://wcbpub.oss-cn-hangzhou.aliyuncs.com/xue/xxxx"

# Test files
TEST_FILES = {
    "pdf": "Effective-harnesses-for-long-running-agents.pdf",
    "docx": "Effective-harnesses-for-long-running-agents.docx",
    "txt": "Effective-harnesses-for-long-running-agents.txt",
    "md": "Effective-harnesses-for-long-running-agents.md",
    "csv": "customers-100.csv",
    "xlsx": "customers-100.xlsx",
}

# Web link for testing
WEB_LINK = "https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents"


def print_result(file_type: str, source: str, result, error=None):
    """Print test result with emoji"""
    prefix = "📄" if source == "local" else "🌐"
    status = "✅" if not error else "❌"
    
    print(f"\n{prefix} {status} [{file_type.upper()}] {source.upper()}")
    
    if error:
        print(f"   Error: {error}")
    elif result and result.content:
        content_preview = result.content[:200].replace('\n', ' ')
        print(f"   📝 Content preview ({len(result.content)} chars): {content_preview}...")
        if result.meta:
            print(f"   ℹ️  Metadata: {result.meta}")
    else:
        print(f"   ⚠️  No content returned")


async def test_pdf_local():
    """Test PDF - Local file"""
    file_path = TMP_DIR / TEST_FILES["pdf"]
    pdf = PDF(str(file_path))
    result = await pdf.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("pdf", "local", result, error if error != "No result" else None)


async def test_pdf_remote():
    """Test PDF - Remote file"""
    url = f"{OSS_BASE_URL}/{TEST_FILES['pdf']}"
    pdf = PDF(url)
    result = await pdf.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("pdf", "remote", result, error if error != "No result" else None)


async def test_doc_local():
    """Test DOCX - Local file"""
    file_path = TMP_DIR / TEST_FILES["docx"]
    doc = DOC(str(file_path))
    result = await doc.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("docx", "local", result, error if error != "No result" else None)


async def test_doc_remote():
    """Test DOCX - Remote file"""
    url = f"{OSS_BASE_URL}/{TEST_FILES['docx']}"
    doc = DOC(url)
    result = await doc.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("docx", "remote", result, error if error != "No result" else None)


async def test_txt_local():
    """Test TXT - Local file"""
    file_path = TMP_DIR / TEST_FILES["txt"]
    txt = TXT(str(file_path))
    result = await txt.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("txt", "local", result, error if error != "No result" else None)


async def test_txt_remote():
    """Test TXT - Remote file"""
    url = f"{OSS_BASE_URL}/{TEST_FILES['txt']}"
    txt = TXT(url)
    result = await txt.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("txt", "remote", result, error if error != "No result" else None)


async def test_md_local():
    """Test Markdown - Local file (using TXT)"""
    file_path = TMP_DIR / TEST_FILES["md"]
    md = TXT(str(file_path))
    result = await md.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("markdown", "local", result, error if error != "No result" else None)


async def test_md_remote():
    """Test Markdown - Remote file (using TXT)"""
    url = f"{OSS_BASE_URL}/{TEST_FILES['md']}"
    md = TXT(url)
    result = await md.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("markdown", "remote", result, error if error != "No result" else None)


async def test_csv_local():
    """Test CSV - Local file"""
    file_path = TMP_DIR / TEST_FILES["csv"]
    csv = CSV(str(file_path))
    result = await csv.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("csv", "local", result, error if error != "No result" else None)


async def test_csv_remote():
    """Test CSV - Remote file"""
    url = f"{OSS_BASE_URL}/{TEST_FILES['csv']}"
    csv = CSV(url)
    result = await csv.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("csv", "remote", result, error if error != "No result" else None)


async def test_excel_local():
    """Test Excel - Local file"""
    file_path = TMP_DIR / TEST_FILES["xlsx"]
    excel = Excel(str(file_path))
    result = await excel.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("excel", "local", result, error if error != "No result" else None)


async def test_excel_remote():
    """Test Excel - Remote file"""
    url = f"{OSS_BASE_URL}/{TEST_FILES['xlsx']}"
    excel = Excel(url)
    result = await excel.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("excel", "remote", result, error if error != "No result" else None)


async def test_link_web():
    """Test Link - Web page"""
    link = Link(WEB_LINK)
    result = await link.aread()
    error = result.meta.get('error') if result else "No result"
    print_result("link", "web", result, error if error != "No result" else None)


async def main():
    """Run all tests"""
    print("=" * 80)
    print("📋 Document Attachment Comprehensive Test")
    print("=" * 80)
    
    # Test PDF
    print("\n" + "─" * 80)
    print("📄 Testing PDF Files")
    print("─" * 80)
    await test_pdf_local()
    await test_pdf_remote()
    
    # Test DOCX
    print("\n" + "─" * 80)
    print("📝 Testing DOCX Files")
    print("─" * 80)
    await test_doc_local()
    await test_doc_remote()
    
    # Test TXT
    print("\n" + "─" * 80)
    print("📃 Testing TXT Files")
    print("─" * 80)
    await test_txt_local()
    await test_txt_remote()
    
    # Test Markdown
    print("\n" + "─" * 80)
    print("📋 Testing Markdown Files")
    print("─" * 80)
    await test_md_local()
    await test_md_remote()
    
    # Test CSV
    print("\n" + "─" * 80)
    print("📊 Testing CSV Files")
    print("─" * 80)
    await test_csv_local()
    await test_csv_remote()
    
    # Test Excel
    print("\n" + "─" * 80)
    print("📈 Testing Excel Files")
    print("─" * 80)
    await test_excel_local()
    await test_excel_remote()
    
    # Test Link
    print("\n" + "─" * 80)
    print("🔗 Testing Web Link")
    print("─" * 80)
    await test_link_web()
    
    print("\n" + "=" * 80)
    print("✨ Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
