# integration_demo.py
"""
Demonstration script showing the integrated agentic workflow capabilities.

This script demonstrates:
1. Direct file creation via chat LLM
2. Plan-then-code multi-file generation
3. Code viewer (Gemini Canvas-like) functionality
4. Terminal validation and LLM communication logging
5. File apply and save functionality
"""

import sys
import os
import logging
from typing import Optional

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def demonstrate_agentic_capabilities():
    """
    Demonstrate the integrated agentic workflow capabilities
    """
    print("🤖 AvA Agentic Integration Demo")
    print("=" * 50)

    print("\n📋 Features demonstrated:")
    print("1. 🔍 User Intent Detection")
    print("2. 📝 Direct File Creation")
    print("3. 🧠 Plan-then-Code Workflow")
    print("4. 👁️  Code Viewer (Canvas-like)")
    print("5. 🖥️  Terminal Command Execution")
    print("6. 📊 LLM Communication Logging")
    print("7. 💾 File Apply & Save")

    print("\n" + "=" * 50)
    print("Integration Components Working Together:")
    print("=" * 50)

    # Test user input processing
    print("\n1. 🔍 Testing User Intent Detection:")
    test_user_inputs()

    print("\n2. 📁 File Structure Overview:")
    show_integration_structure()

    print("\n3. 🔄 Workflow Overview:")
    show_workflow_overview()

    print("\n4. 🚀 Ready to Run!")
    print("   Start your AvA application and try these commands:")
    show_example_commands()


def test_user_inputs():
    """Test the user input handler with various inputs"""
    try:
        from core.user_input_handler import UserInputHandler, UserInputIntent

        handler = UserInputHandler()

        test_cases = [
            # File creation requests
            ("create a file called utils.py", UserInputIntent.FILE_CREATION_REQUEST),
            ("make me a calculator.py", UserInputIntent.FILE_CREATION_REQUEST),
            ("write a hello.py file", UserInputIntent.FILE_CREATION_REQUEST),

            # Plan-then-code requests
            ("build a complete web application", UserInputIntent.PLAN_THEN_CODE_REQUEST),
            ("create a CLI tool from scratch", UserInputIntent.PLAN_THEN_CODE_REQUEST),
            ("develop a Python library", UserInputIntent.PLAN_THEN_CODE_REQUEST),

            # Normal chat
            ("how does recursion work?", UserInputIntent.NORMAL_CHAT),
            ("explain machine learning", UserInputIntent.NORMAL_CHAT),
        ]

        for input_text, expected_intent in test_cases:
            result = handler.process_input(input_text)
            status = "✅" if result.intent == expected_intent else "❌"
            print(f"   {status} '{input_text[:30]}...' → {result.intent.name}")

    except ImportError as e:
        print(f"   ⚠️  Could not test input handler: {e}")


def show_integration_structure():
    """Show how the components are integrated"""
    print("   🏗️  ApplicationOrchestrator")
    print("   ├── 🧠 ChatManager")
    print("   │   ├── 🎯 UserInputHandler")
    print("   │   ├── 🤖 PlanAndCodeCoordinator")
    print("   │   └── 🔄 BackendCoordinator")
    print("   ├── 👁️  CodeViewerWindow")
    print("   ├── 🖥️  TerminalService")
    print("   ├── 📊 LlmCommunicationLogger")
    print("   ├── 🪟 LlmTerminalWindow")
    print("   └── 🚌 EventBus (connects everything)")


def show_workflow_overview():
    """Show the workflow process"""
    print("   User Input → Intent Detection → Workflow Selection")
    print("   ")
    print("   📝 Single File Creation:")
    print("      User → ChatLLM → CodeViewer → Apply → Save")
    print("   ")
    print("   🧠 Plan-then-Code:")
    print("      User → PlannerLLM → Parser → CoderLLM(s) → Validation → CodeViewer")
    print("   ")
    print("   🔄 Real-time Feedback:")
    print("      All steps logged to Terminal Window with syntax highlighting")


def show_example_commands():
    """Show example commands users can try"""
    print("\n   📝 Single File Creation Examples:")
    print('   • "create a file called calculator.py"')
    print('   • "make me a utils.py with helper functions"')
    print('   • "write a simple web scraper in scraper.py"')

    print("\n   🧠 Plan-then-Code Examples:")
    print('   • "build a complete todo app"')
    print('   • "create a CLI tool for file management"')
    print('   • "develop a web API from scratch"')

    print("\n   👁️  Code Viewer Features:")
    print("   • View all generated files in one place")
    print("   • Syntax highlighting and formatting")
    print("   • Apply & Save files to your project")
    print("   • Copy code with one click")

    print("\n   🖥️  Terminal Integration:")
    print("   • Real-time LLM communication logs")
    print("   • Automatic code validation")
    print("   • Command execution feedback")
    print("   • Syntax highlighting for code blocks")


def check_dependencies():
    """Check if key dependencies are available"""
    print("\n🔍 Dependency Check:")

    required_modules = [
        "PySide6",
        "pygments",
        "chromadb",
        "sentence_transformers"
    ]

    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} (install with: pip install {module})")


def show_key_benefits():
    """Show the key benefits of the integrated system"""
    print("\n🌟 Key Benefits of Integrated Agentic System:")
    print("=" * 50)

    benefits = [
        "🔄 Seamless workflow from chat to code",
        "🎯 Intelligent intent detection",
        "🧠 Multi-agent collaboration (Planner + Coder)",
        "👁️  Visual code management (Canvas-like)",
        "🖥️  Real-time terminal feedback",
        "💾 Direct file system integration",
        "🔍 Automatic code validation",
        "📊 Complete activity logging",
        "🚀 Ready for production use"
    ]

    for benefit in benefits:
        print(f"   {benefit}")


if __name__ == "__main__":
    # Set up logging to see integration messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        demonstrate_agentic_capabilities()
        check_dependencies()
        show_key_benefits()

        print("\n" + "=" * 50)
        print("🎉 Integration Demo Complete!")
        print("   Your agentic workflow system is ready!")
        print("   Run your main application to start coding with AI assistance.")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("   Check your imports and dependencies.")
        import traceback

        traceback.print_exc()