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
        st.markdown("**Current Model:** Phi-2 Demo")
        st.markdown("**Status:** Ready for Chain of Thought reasoning")
        
        if st.button("🔄 Initialize Model", type="primary"):
            with st.spinner("Initializing Phi-2 model..."):
                try:
                    st.session_state.model_manager = ModelManager(
                        model_name="microsoft/phi-2",
                        use_quantization=True
                    )
                    st.session_state.cot_generator = CoTGenerator(
                        st.session_state.model_manager,
                        max_length=300
                    )
                    st.success("✅ Phi-2 model ready!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.session_state.model_manager = None
                    st.session_state.cot_generator = None
    
    # Main content area
    if st.session_state.model_manager is None:
        st.info("👈 Please initialize the Phi-2 model from the sidebar to get started.")
        st.markdown("### About Chain of Thought Reasoning")
        st.markdown("""
        Chain of Thought (CoT) prompting is a technique that encourages language models to:
        - Break down complex problems into smaller steps
        - Show intermediate reasoning processes
        - Arrive at answers through logical progression
        
        This application demonstrates how Phi-2 thinks through problems step by step.
        """)
        return
    
    # Input section
    st.header("📝 Problem Input")
    
    problem_to_solve = st.text_area(
        "Enter your problem:",
        value="If a train leaves the station at 3 PM traveling at 60 mph and needs to cover 180 miles, what time will it arrive?",
        height=100,
        help="Modify this problem or enter your own"
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
                
                # Display results in tabs
                tab1, tab2 = st.tabs(["💭 Reasoning Steps", "📊 Flow Visualization"])
                
                with tab1:
                    st.session_state.step_visualizer.display_steps(steps)
                    
                    # Show evolution progress
                    st.subheader("🔄 Reasoning Evolution")
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(100)  # Show completed progress
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Steps", len(steps))
                        with col2:
                            step_types = [step.get('type', 'reasoning') for step in steps]
                            most_common = max(set(step_types), key=step_types.count) if step_types else "reasoning"
                            st.metric("Primary Type", most_common.title())
                        with col3:
                            st.metric("Completion", "100%")
                    
                    # Final LLM response
                    st.subheader("🤖 Final Phi-2 Response")
                    model_info = st.session_state.model_manager.get_model_info()
                    st.info(f"**Model:** {model_info['name']} | **Device:** {model_info['device']}")
                    with st.expander("View raw model output", expanded=False):
                        st.text_area("Complete reasoning output:", value=cot_response, height=150, disabled=True)
                
                with tab2:
                    try:
                        flowchart = st.session_state.flowchart_generator.create_flowchart(steps)
                        st.plotly_chart(flowchart, use_container_width=True)
                        
                        # Additional visualization
                        st.subheader("📈 Step Distribution")
                        distribution = st.session_state.flowchart_generator.create_step_distribution(steps)
                        st.plotly_chart(distribution, use_container_width=True)
                        
                    except Exception as e:
                        st.info("Flowchart visualization not available")
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.info("Try adjusting the model settings or rephrasing your problem.")

if __name__ == "__main__":
    main()
