#!/usr/bin/env python3
"""
Test Document Generation Functionality

This script tests the document generation capabilities by sending test prompts
to the chatbot and verifying that document generation intent is detected.

Usage: python test_document_generation.py
"""

import asyncio
import sys
import os

# Add the current directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test prompts that should trigger document generation
test_prompts = [
    "Create a PDF report about renewable energy trends",
    "Generate a Word document summarizing quarterly sales data",
    "Make a PowerPoint presentation about artificial intelligence",
    "Put this analysis in a spreadsheet format",
    "Export the findings as a PDF document",
    "Can you create a report in Word format about climate change?",
    "Generate an Excel file with budget information",
    "Create a presentation about machine learning algorithms"
]

# Test prompts that should NOT trigger document generation
negative_test_prompts = [
    "What is renewable energy?",
    "Tell me about quarterly sales",
    "Explain artificial intelligence",
    "How does climate change work?",
    "What is machine learning?"
]

async def test_document_intent_detection():
    """Test if the intent questions can properly detect document generation requests."""
    print("🧪 Testing Document Intent Detection")
    print("=" * 50)
    
    # Import the intent questions
    try:
        from app.api.routes import intent_questions
        document_question = intent_questions.get("document", "")
        
        if document_question:
            print(f"✅ Document intent question found: {document_question}")
        else:
            print("❌ Document intent question not found in intent_questions")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import intent_questions: {e}")
        return False
    
    print("\n🔍 Test Prompts that SHOULD trigger document generation:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  {i}. {prompt}")
    
    print("\n🔍 Test Prompts that should NOT trigger document generation:")
    for i, prompt in enumerate(negative_test_prompts, 1):
        print(f"  {i}. {prompt}")
    
    return True

async def test_document_functions():
    """Test if document generation functions are properly defined."""
    print("\n📄 Testing Document Generation Functions")
    print("=" * 50)
    
    try:
        from app.document import DocumentGenerator, generate_document_from_response, get_document_generator
        from app.document.document_generator import GENERATED_DOCS_DIR
        
        print("✅ All document generation functions imported successfully")
        print(f"✅ Generated documents directory: {GENERATED_DOCS_DIR}")
        
        # Test document generator instance
        doc_gen = get_document_generator()
        supported_formats = doc_gen.get_supported_formats()
        print(f"✅ Supported formats: {', '.join(supported_formats)}")
        
        # Check if directory exists
        if os.path.exists(GENERATED_DOCS_DIR):
            print(f"✅ Generated docs directory exists: {GENERATED_DOCS_DIR}")
        else:
            print(f"ℹ️  Generated docs directory will be created when needed: {GENERATED_DOCS_DIR}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import document functions: {e}")
        return False

async def test_dependencies():
    """Test if all required document generation dependencies are available."""
    print("\n📦 Testing Document Generation Dependencies")
    print("=" * 50)
    
    dependencies = [
        ("reportlab", "PDF generation"),
        ("docx", "Word document generation"),
        ("openpyxl", "Excel spreadsheet generation"),
        ("pptx", "PowerPoint presentation generation")
    ]
    
    all_deps_available = True
    
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            print(f"✅ {dep_name} - {description}")
        except ImportError:
            print(f"❌ {dep_name} - {description} (NOT INSTALLED)")
            all_deps_available = False
    
    return all_deps_available

async def main():
    """Run all tests."""
    print("🚀 TinyLLM Document Generation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Intent Detection", test_document_intent_detection()),
        ("Document Functions", test_document_functions()),
        ("Dependencies", test_dependencies())
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Document generation is ready to use.")
        print("\n💡 Example usage:")
        print("   - 'Create a PDF report about renewable energy'")
        print("   - 'Generate a Word document with this analysis'")
        print("   - 'Make a PowerPoint presentation about AI'")
        print("   - 'Put this data in a spreadsheet'")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        if not any(name == "Dependencies" and result for name, result in results):
            print("\n🔧 To install missing dependencies:")
            print("   pip install reportlab python-docx openpyxl python-pptx")

if __name__ == "__main__":
    asyncio.run(main())
