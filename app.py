import streamlit as st
import sys
import os
import time

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model import ModelManager
from core.cot_generator import CoTGenerator
from core.examples import ExampleProblems
from visualization.step_display import StepVisualizer
from visualization.flowchart import FlowchartGenerator

def check_auth_token():
    """Check if authentication token is properly configured."""
    from dotenv import load_dotenv
    load_dotenv()
    
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token or token == 'your_token_here':
        return False, "Token not configured"
    return True, f"Token: {token[:10]}...{token[-10:] if len(token) > 20 else '***'}"

def main():
    st.set_page_config(
        page_title="Chain of Thought Visualizer",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Chain of Thought Reasoning Visualizer")
    st.markdown("Explore how language models think step-by-step through complex problems using **real Phi-2 model**")
    
    # Initialize session state
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'cot_generator' not in st.session_state:
        st.session_state.cot_generator = None
    if 'step_visualizer' not in st.session_state:
        st.session_state.step_visualizer = StepVisualizer()
    if 'flowchart_generator' not in st.session_state:
        st.session_state.flowchart_generator = FlowchartGenerator()
    if 'model_loading' not in st.session_state:
        st.session_state.model_loading = False
    
    # Check authentication first
    auth_ok, auth_status = check_auth_token()
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Authentication status
        if auth_ok:
            st.success("✅ Authentication Ready")
            st.caption(auth_status)
        else:
            st.error("❌ Authentication Required")
            st.markdown("""
            **Setup Required:**
            1. Get your token from [Hugging Face](https://huggingface.co/settings/tokens)
            2. Update the `.env` file with your token
            3. Restart the application
            """)
            st.stop()
        
        # Model status display
        if st.session_state.model_manager and st.session_state.model_manager.is_ready():
            model_info = st.session_state.model_manager.get_model_info()
            st.success("✅ Model Ready")
            st.markdown(f"**Model:** {model_info['model_type']}")
            st.markdown(f"**Device:** {model_info['device']}")
            st.markdown(f"**Parameters:** {model_info['total_parameters']}")
            if model_info['quantized']:
                st.info("🔧 Using 4-bit quantization")
        else:
            st.warning("⚠️ Model Not Loaded")
            st.markdown("Click 'Initialize Model' to load Phi-2")
        
        st.markdown("---")
        
        # Model configuration options
        st.subheader("🔧 Configuration")
        
        use_quantization = st.checkbox(
            "Use 4-bit Quantization", 
            value=True, 
            help="Reduces memory usage but may slightly affect performance"
        )
        
        max_length = st.slider(
            "Max Response Length", 
            min_value=100, 
            max_value=2000, 
            value=1024, 
            step=100,
            help="Maximum number of tokens in the response"
        )
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.1, 
            max_value=1.5, 
            value=0.3, 
            step=0.1,
            help="Controls randomness (higher = more creative, lower = more focused)"
        )
        
        top_p = st.slider(
            "Top-p (Nucleus Sampling)", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.9, 
            step=0.05,
            help="Controls diversity of responses"
        )
        
        # Initialize model button
        if st.button("🚀 Initialize Phi-2 Model", type="primary", disabled=st.session_state.model_loading):
            if not st.session_state.model_loading:
                st.session_state.model_loading = True
                
                with st.spinner("Loading Phi-2 model (this may take a few minutes)..."):
                    try:
                        # Clean up existing model if any
                        if st.session_state.model_manager:
                            st.session_state.model_manager.cleanup()
                        
                        # Initialize new model
                        st.session_state.model_manager = ModelManager(
                            model_name="microsoft/phi-2",
                            use_quantization=use_quantization
                        )
                        
                        # Initialize CoT generator
                        st.session_state.cot_generator = CoTGenerator(
                            st.session_state.model_manager,
                            max_length=max_length
                        )
                        
                        st.success("✅ Phi-2 model loaded successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
                        st.session_state.model_manager = None
                        st.session_state.cot_generator = None
                    finally:
                        st.session_state.model_loading = False
                        st.rerun()
        
        # Cleanup button
        if st.session_state.model_manager and st.session_state.model_manager.is_ready():
            if st.button("🗑️ Unload Model", type="secondary"):
                st.session_state.model_manager.cleanup()
                st.session_state.model_manager = None
                st.session_state.cot_generator = None
                st.success("Model unloaded successfully!")
                st.rerun()
        
        st.markdown("---")
        
        # System requirements
        st.subheader("💻 System Info")
        import torch
        if torch.cuda.is_available():
            st.success("✅ CUDA Available")
            st.markdown(f"GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            st.info("🍎 MPS (Apple Silicon) Available")
        else:
            st.warning("⚠️ CPU Only (slower)")
    
    # Main content area
    if st.session_state.model_manager is None or not st.session_state.model_manager.is_ready():
        st.info("👈 Please initialize the Phi-2 model from the sidebar to get started.")
        
        # Show model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 About Chain of Thought Reasoning")
            st.markdown("""
            Chain of Thought (CoT) prompting is a technique that encourages language models to:
            - Break down complex problems into smaller steps
            - Show intermediate reasoning processes
            - Arrive at answers through logical progression
            
            This application uses the **real Microsoft Phi-2 model** to demonstrate actual AI reasoning.
            """)
        
        with col2:
            st.markdown("### 🔧 Technical Details")
            st.markdown("""
            **Model:** Microsoft Phi-2 (2.7B parameters)
            **Framework:** Hugging Face Transformers
            **Hardware:** Automatic GPU/CPU detection
            **Optimization:** Optional 4-bit quantization
            
            The model will generate real, dynamic reasoning chains for your problems.
            """)
        
        # Show example problems
        st.markdown("### 📝 Example Problems")
        examples = ExampleProblems()
        
        tab1, tab2, tab3 = st.tabs(["🧮 Math", "🧩 Logic", "🤔 Riddles"])
        
        with tab1:
            for i, problem in enumerate(examples.get_math_problems()[:3]):
                st.markdown(f"**{i+1}.** {problem['question']}")
        
        with tab2:
            for i, problem in enumerate(examples.get_logic_problems()[:3]):
                st.markdown(f"**{i+1}.** {problem['question']}")
        
        with tab3:
            for i, problem in enumerate(examples.get_riddles()[:3]):
                st.markdown(f"**{i+1}.** {problem['question']}")
        
        return
    
    # Input section
    st.header("📝 Problem Input")
    
    # Problem input with example selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        problem_to_solve = st.text_area(
            "Enter your problem:",
            value="If a train leaves the station at 3 PM traveling at 60 mph and needs to cover 180 miles, what time will it arrive?",
            height=100,
            help="Modify this problem or enter your own"
        )
    
    with col2:
        st.markdown("**Or try an example:**")
        examples = ExampleProblems()
        
        if st.button("🧮 Math Problem"):
            problem = examples.get_random_problem("math")
            st.session_state.example_problem = problem['question']
            st.rerun()
        
        if st.button("🧩 Logic Problem"):
            problem = examples.get_random_problem("logic")
            st.session_state.example_problem = problem['question']
            st.rerun()
        
        if st.button("🤔 Riddle"):
            problem = examples.get_random_problem("riddle")
            st.session_state.example_problem = problem['question']
            st.rerun()
    
    # Update problem if example was selected
    if 'example_problem' in st.session_state:
        problem_to_solve = st.session_state.example_problem
        del st.session_state.example_problem
    
    if problem_to_solve and st.button("🚀 Generate Chain of Thought", type="primary"):
        if st.session_state.cot_generator is None:
            st.error("Please load a model first!")
            return
            
        st.header("🔍 Chain of Thought Analysis")
        
        # Generation progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Update progress
            progress_bar.progress(25)
            status_text.text("Analyzing problem type...")
            
            # Generate CoT response
            progress_bar.progress(50)
            status_text.text("Generating reasoning with Phi-2...")
            
            start_time = time.time()
            cot_response = st.session_state.cot_generator.generate_cot(
                problem_to_solve,
                temperature=temperature,
                top_p=top_p
            )
            generation_time = time.time() - start_time
            
            progress_bar.progress(75)
            status_text.text("Parsing reasoning steps...")
            
            # Parse steps
            steps = st.session_state.cot_generator.parse_steps(cot_response)
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["💭 Reasoning Steps", "📊 Flow Visualization", "🔧 Generation Info"])
            
            with tab1:
                st.session_state.step_visualizer.display_steps(steps)
                
                # Show generation metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Generation Time", f"{generation_time:.2f}s")
                with col2:
                    st.metric("Total Steps", len(steps))
                with col3:
                    step_types = [step.get('type', 'reasoning') for step in steps]
                    most_common = max(set(step_types), key=step_types.count) if step_types else "reasoning"
                    st.metric("Primary Type", most_common.title())
                with col4:
                    st.metric("Response Length", f"{len(cot_response)} chars")
                
                # Final response
                st.subheader("🤖 Complete Phi-2 Response")
                model_info = st.session_state.model_manager.get_model_info()
                st.info(f"**Model:** {model_info['model_type']} | **Device:** {model_info['device']} | **Parameters:** {model_info['total_parameters']}")
                
                with st.expander("View raw model output", expanded=False):
                    st.text_area("Complete reasoning output:", value=cot_response, height=200, disabled=True)
            
            with tab2:
                try:
                    flowchart = st.session_state.flowchart_generator.create_flowchart(steps)
                    st.plotly_chart(flowchart, use_container_width=True)
                    
                    # Additional visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Step Distribution")
                        distribution = st.session_state.flowchart_generator.create_step_distribution(steps)
                        st.plotly_chart(distribution, use_container_width=True)
                    
                    with col2:
                        st.subheader("🕐 Step Timeline")
                        timeline = st.session_state.flowchart_generator.create_step_timeline(steps)
                        st.plotly_chart(timeline, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error creating visualizations: {str(e)}")
                    st.info("Flowchart visualization not available")
            
            with tab3:
                st.subheader("🔧 Generation Configuration")
                
                # Show current settings
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Model Settings:**")
                    st.markdown(f"- Temperature: {temperature}")
                    st.markdown(f"- Top-p: {top_p}")
                    st.markdown(f"- Max Length: {max_length}")
                    st.markdown(f"- Quantization: {'Yes' if use_quantization else 'No'}")
                
                with col2:
                    st.markdown("**Performance Metrics:**")
                    st.markdown(f"- Generation Time: {generation_time:.2f}s")
                    st.markdown(f"- Response Length: {len(cot_response)} characters")
                    st.markdown(f"- Steps Parsed: {len(steps)}")
                    st.markdown(f"- Model Device: {model_info['device']}")
                
                # Show generation stats
                stats = st.session_state.cot_generator.get_generation_stats()
                st.subheader("📊 Model Statistics")
                st.json(stats)
                
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            st.info("Try adjusting the model settings or rephrasing your problem.")

if __name__ == "__main__":
    main()
