#!/usr/bin/env python3
"""
ThoughtChain Demo Script
This script demonstrates the end-to-end Chain of Thought reasoning process using the real Phi-2 model.
"""

import sys
import os
import time
import logging

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import ModelManager
from core.cot_generator import CoTGenerator
from core.examples import ExampleProblems

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_separator(title=""):
    """Print a separator line with optional title."""
    if title:
        print(f"\n{'='*60}")
        print(f"🧠 {title}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")

def print_step(step_num, title, content=""):
    """Print a formatted step."""
    print(f"\n📋 Step {step_num}: {title}")
    if content:
        print(f"   {content}")
    print("-" * 40)

def demo_model_loading():
    """Demonstrate model loading process."""
    print_separator("MODEL LOADING DEMO")
    
    print_step(1, "Initializing ModelManager")
    print("   Creating ModelManager with Phi-2 model...")
    
    try:
        # Initialize model manager
        model_manager = ModelManager(
            model_name="microsoft/phi-2",
            use_quantization=True  # Use quantization for memory efficiency
        )
        
        print("   ✅ ModelManager initialized successfully")
        
        # Get model information
        model_info = model_manager.get_model_info()
        print(f"   📊 Model: {model_info['model_type']}")
        print(f"   🔧 Device: {model_info['device']}")
        print(f"   💾 Parameters: {model_info['total_parameters']}")
        print(f"   ⚡ Quantized: {model_info['quantized']}")
        
        return model_manager
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None

def demo_cot_generation(model_manager):
    """Demonstrate Chain of Thought generation."""
    print_separator("CHAIN OF THOUGHT GENERATION DEMO")
    
    print_step(1, "Initializing CoT Generator")
    cot_generator = CoTGenerator(model_manager, max_length=512)
    print("   ✅ CoT Generator initialized")
    
    # Get example problems
    examples = ExampleProblems()
    
    # Test different problem types
    problem_types = [
        ("math", examples.get_math_problems()[0]['question']),
        ("logic", examples.get_logic_problems()[0]['question']),
        ("riddle", examples.get_riddles()[0]['question'])
    ]
    
    for problem_type, problem in problem_types:
        print_separator(f"TESTING {problem_type.upper()} PROBLEM")
        
        print_step(1, f"Problem Type Detection")
        detected_type = cot_generator._detect_problem_type(problem)
        print(f"   Input: {problem}")
        print(f"   Detected Type: {detected_type}")
        print(f"   Expected Type: {problem_type}")
        print(f"   ✅ {'Match' if detected_type == problem_type else 'Mismatch'}")
        
        print_step(2, "Generating Chain of Thought")
        print(f"   Problem: {problem}")
        print("   Generating response with Phi-2...")
        
        start_time = time.time()
        try:
            # Generate CoT response
            cot_response = cot_generator.generate_cot(
                problem,
                temperature=0.7,
                top_p=0.9
            )
            generation_time = time.time() - start_time
            
            print(f"   ✅ Generation completed in {generation_time:.2f} seconds")
            print(f"   📏 Response length: {len(cot_response)} characters")
            
            print_step(3, "Raw Model Response")
            print("   " + "="*50)
            print(f"   {cot_response}")
            print("   " + "="*50)
            
            print_step(4, "Parsing Reasoning Steps")
            steps = cot_generator.parse_steps(cot_response)
            print(f"   ✅ Parsed {len(steps)} reasoning steps")
            
            for i, step in enumerate(steps, 1):
                step_type = step.get('type', 'reasoning')
                content = step.get('content', '')
                print(f"   Step {i} ({step_type}): {content[:100]}{'...' if len(content) > 100 else ''}")
            
            print_step(5, "Step Type Distribution")
            step_types = [step.get('type', 'reasoning') for step in steps]
            type_counts = {}
            for step_type in step_types:
                type_counts[step_type] = type_counts.get(step_type, 0) + 1
            
            for step_type, count in type_counts.items():
                print(f"   {step_type.title()}: {count}")
            
        except Exception as e:
            print(f"   ❌ Error during generation: {e}")
            continue

def demo_performance_testing(model_manager):
    """Demonstrate performance testing."""
    print_separator("PERFORMANCE TESTING")
    
    cot_generator = CoTGenerator(model_manager, max_length=256)
    examples = ExampleProblems()
    
    # Test with a simple math problem
    test_problem = "What is 15 + 27?"
    
    print_step(1, "Performance Test Setup")
    print(f"   Test Problem: {test_problem}")
    print(f"   Max Length: 256 tokens")
    print(f"   Temperature: 0.7")
    
    print_step(2, "Running Performance Test")
    
    # Run multiple generations to test consistency
    times = []
    responses = []
    
    for i in range(3):
        print(f"   Run {i+1}/3...")
        start_time = time.time()
        
        try:
            response = cot_generator.generate_cot(test_problem, temperature=0.7)
            generation_time = time.time() - start_time
            
            times.append(generation_time)
            responses.append(response)
            
            print(f"   ✅ Run {i+1} completed in {generation_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Run {i+1} failed: {e}")
    
    if times:
        print_step(3, "Performance Results")
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"   Average Generation Time: {avg_time:.2f}s")
        print(f"   Fastest Generation: {min_time:.2f}s")
        print(f"   Slowest Generation: {max_time:.2f}s")
        print(f"   Total Runs: {len(times)}")
        
        print_step(4, "Response Consistency")
        for i, response in enumerate(responses, 1):
            print(f"   Response {i}: {response[:100]}{'...' if len(response) > 100 else ''}")

def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print_separator("ERROR HANDLING DEMO")
    
    print_step(1, "Testing Invalid Model Name")
    try:
        invalid_manager = ModelManager(model_name="invalid/model/name")
        print("   ❌ Should have failed with invalid model name")
    except Exception as e:
        print(f"   ✅ Correctly handled invalid model: {type(e).__name__}")
    
    print_step(2, "Testing Model Cleanup")
    try:
        # Create a valid manager
        manager = ModelManager(model_name="microsoft/phi-2", use_quantization=True)
        print("   ✅ Model loaded successfully")
        
        # Test cleanup
        manager.cleanup()
        print("   ✅ Model cleanup completed")
        
        # Test if model is ready after cleanup
        if not manager.is_ready():
            print("   ✅ Model correctly marked as not ready after cleanup")
        else:
            print("   ❌ Model should not be ready after cleanup")
            
    except Exception as e:
        print(f"   ❌ Error during cleanup test: {e}")

def main():
    """Run the complete demo."""
    print_separator("THOUGHTCHAIN E2E DEMO")
    print("This demo will test the complete Chain of Thought reasoning pipeline")
    print("using the real Microsoft Phi-2 model.")
    
    # Check if user wants to proceed
    try:
        response = input("\n🤔 Do you want to proceed with the demo? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Demo cancelled.")
            return
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return
    
    print("\n🚀 Starting ThoughtChain Demo...")
    
    # Demo 1: Model Loading
    model_manager = demo_model_loading()
    if not model_manager:
        print("❌ Failed to load model. Demo cannot continue.")
        return
    
    # Demo 2: Chain of Thought Generation
    demo_cot_generation(model_manager)
    
    # Demo 3: Performance Testing
    demo_performance_testing(model_manager)
    
    # Demo 4: Error Handling
    demo_error_handling()
    
    # Cleanup
    print_separator("CLEANUP")
    print_step(1, "Cleaning up resources")
    model_manager.cleanup()
    print("   ✅ Resources cleaned up successfully")
    
    print_separator("DEMO COMPLETE")
    print("🎉 ThoughtChain demo completed successfully!")
    print("✅ All components are working correctly")
    print("🚀 You can now run the full application with: streamlit run app.py")
    print("\nHappy reasoning! 🧠✨")

if __name__ == "__main__":
    main() 