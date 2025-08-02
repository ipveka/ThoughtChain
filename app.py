import streamlit as st
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model import ModelManager
from core.cot_generator import CoTGenerator
from core.examples import ExampleProblems
from visualization.step_display import StepVisualizer
from visualization.flowchart import FlowchartGenerator

def main():
    st.set_page_config(
        page_title="Chain of Thought Visualizer",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Chain of Thought Reasoning Visualizer")
    st.markdown("Explore how language models think step-by-step through complex problems")
    
    # Initialize session state
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'cot_generator' not in st.session_state:
        st.session_state.cot_generator = None
    if 'step_visualizer' not in st.session_state:
        st.session_state.step_visualizer = StepVisualizer()
    if 'flowchart_generator' not in st.session_state:
        st.session_state.flowchart_generator = FlowchartGenerator()
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        model_options = [
            "Simulated-CoT-Model",
            "Educational-Reasoning-Model",
            "Demo-Chain-of-Thought"
        ]
        
        selected_model = st.selectbox(
            "Choose a model:",
            model_options,
            help="Select a lightweight model for local inference"
        )
        
        use_quantization = st.checkbox(
            "Use quantization (recommended for CPU)",
            value=True,
            help="Reduces memory usage and improves CPU performance"
        )
        
        max_length = st.slider(
            "Max response length:",
            min_value=50,
            max_value=500,
            value=200,
            help="Maximum number of tokens in the response"
        )
        
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model... This may take a few minutes."):
                try:
                    st.session_state.model_manager = ModelManager(
                        model_name=selected_model,
                        use_quantization=use_quantization
                    )
                    st.session_state.cot_generator = CoTGenerator(
                        st.session_state.model_manager,
                        max_length=max_length
                    )
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.session_state.model_manager = None
                    st.session_state.cot_generator = None
    
    # Main content area
    if st.session_state.model_manager is None:
        st.info("👈 Please load a model from the sidebar to get started.")
        st.markdown("### About Chain of Thought Reasoning")
        st.markdown("""
        Chain of Thought (CoT) prompting is a technique that encourages language models to:
        - Break down complex problems into smaller steps
        - Show intermediate reasoning processes
        - Arrive at answers through logical progression
        
        This application demonstrates how models think through different types of problems step by step.
        """)
        return
    
    # Input section
    st.header("📝 Problem Input")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["Custom Problem", "Example Problems"])
    
    with tab1:
        user_input = st.text_area(
            "Enter your problem:",
            placeholder="e.g., If a train leaves at 3 PM traveling at 60 mph and needs to cover 180 miles, what time will it arrive?",
            height=100
        )
    
    with tab2:
        example_problems = ExampleProblems()
        category = st.selectbox(
            "Choose category:",
            ["Math Problems", "Logic Puzzles", "Riddles"]
        )
        
        if category == "Math Problems":
            examples = example_problems.get_math_problems()
        elif category == "Logic Puzzles":
            examples = example_problems.get_logic_problems()
        else:
            examples = example_problems.get_riddles()
        
        selected_example = st.selectbox(
            "Choose an example:",
            [ex["question"] for ex in examples]
        )
        
        if st.button("📋 Use This Example"):
            user_input = selected_example
            st.rerun()
    
    # Processing section
    problem_to_solve = user_input if 'user_input' in locals() and user_input else (
        selected_example if 'selected_example' in locals() else ""
    )
    
    if problem_to_solve and st.button("🚀 Generate Chain of Thought", type="primary"):
        if st.session_state.cot_generator is None:
            st.error("Please load a model first!")
            return
            
        st.header("🔍 Chain of Thought Analysis")
        
        with st.spinner("Generating step-by-step reasoning..."):
            try:
                # Generate CoT response
                cot_response = st.session_state.cot_generator.generate_cot(problem_to_solve)
                
                # Parse steps
                steps = st.session_state.cot_generator.parse_steps(cot_response)
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("💭 Reasoning Steps")
                    st.session_state.step_visualizer.display_steps(steps)
                
                with col2:
                    st.subheader("📊 Flow Visualization")
                    try:
                        flowchart = st.session_state.flowchart_generator.create_flowchart(steps)
                        st.plotly_chart(flowchart, use_container_width=True)
                    except Exception as e:
                        st.info("Flowchart visualization not available")
                
                # Raw response section (collapsible)
                with st.expander("🔧 Raw Model Response"):
                    st.text(cot_response)
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.info("Try adjusting the model settings or rephrasing your problem.")

if __name__ == "__main__":
    main()
