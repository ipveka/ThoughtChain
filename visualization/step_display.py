import streamlit as st
from typing import List, Dict

class StepVisualizer:
    """Handles the display of Chain of Thought reasoning steps in the Streamlit UI."""
    
    def __init__(self):
        self.step_icons = {
            "reasoning": "🤔",
            "calculation": "🧮",
            "conclusion": "✅",
            "assumption": "📝"
        }
        
        self.step_colors = {
            "reasoning": "#E3F2FD",
            "calculation": "#FFF3E0",
            "conclusion": "#E8F5E8",
            "assumption": "#F3E5F5"
        }
    
    def display_steps(self, steps: List[Dict[str, str]]):
        """Display the reasoning steps in an organized format."""
        if not steps:
            st.warning("No reasoning steps found.")
            return
        
        # Display each step
        for i, step in enumerate(steps):
            self._display_single_step(step, i + 1)
        
        # Summary section
        if len(steps) > 1:
            self._display_summary(steps)
    
    def _display_single_step(self, step: Dict[str, str], display_number: int):
        """Display a single reasoning step."""
        step_type = step.get("type", "reasoning")
        icon = self.step_icons.get(step_type, "💭")
        content = step.get("content", "")
        
        # Create an expandable section for each step
        with st.container():
            # Step header
            st.markdown(f"### {icon} Step {display_number}")
            
            # Step content in a nice container
            with st.container():
                st.markdown(f"**Type:** {step_type.title()}")
                st.markdown(content)
            
            # Add some spacing
            st.markdown("---")
    
    def _display_summary(self, steps: List[Dict[str, str]]):
        """Display a summary of the reasoning process."""
        st.subheader("📋 Reasoning Summary")
        
        # Count step types
        step_counts = {}
        for step in steps:
            step_type = step.get("type", "reasoning")
            step_counts[step_type] = step_counts.get(step_type, 0) + 1
        
        # Display metrics
        cols = st.columns(len(step_counts))
        for i, (step_type, count) in enumerate(step_counts.items()):
            with cols[i]:
                icon = self.step_icons.get(step_type, "💭")
                st.metric(
                    label=f"{icon} {step_type.title()}",
                    value=count
                )
        
        # Display conclusion if available
        conclusion_steps = [s for s in steps if s.get("type") == "conclusion"]
        if conclusion_steps:
            st.subheader("🎯 Final Answer")
            st.success(conclusion_steps[-1].get("content", ""))
    
    def display_compact_steps(self, steps: List[Dict[str, str]]):
        """Display steps in a more compact format."""
        if not steps:
            st.warning("No reasoning steps found.")
            return
        
        for i, step in enumerate(steps):
            step_type = step.get("type", "reasoning")
            icon = self.step_icons.get(step_type, "💭")
            content = step.get("content", "")
            
            # Use columns for compact display
            col1, col2 = st.columns([1, 8])
            
            with col1:
                st.markdown(f"**{icon}**")
            
            with col2:
                st.markdown(f"**Step {i + 1}:** {content}")
    
    def display_timeline_steps(self, steps: List[Dict[str, str]]):
        """Display steps in a timeline format."""
        if not steps:
            st.warning("No reasoning steps found.")
            return
        
        st.subheader("🕐 Reasoning Timeline")
        
        for i, step in enumerate(steps):
            step_type = step.get("type", "reasoning")
            icon = self.step_icons.get(step_type, "💭")
            content = step.get("content", "")
            
            # Timeline entry
            with st.container():
                col1, col2, col3 = st.columns([1, 1, 8])
                
                with col1:
                    st.markdown(f"**{i + 1}**")
                
                with col2:
                    st.markdown(f"{icon}")
                
                with col3:
                    st.markdown(content)
                
                # Add connector line (except for last step)
                if i < len(steps) - 1:
                    st.markdown("&nbsp;&nbsp;&nbsp;|")
