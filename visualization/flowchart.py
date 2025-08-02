import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FlowchartGenerator:
    """Generates flowchart visualizations for Chain of Thought reasoning steps."""
    
    def __init__(self):
        self.step_colors = {
            "reasoning": "#81C784",    # Light green
            "calculation": "#FFB74D",  # Light orange
            "conclusion": "#64B5F6",   # Light blue
            "assumption": "#BA68C8"    # Light purple
        }
        
        self.default_color = "#9E9E9E"  # Gray
    
    def create_flowchart(self, steps: List[Dict[str, str]]):
        """Create a flowchart visualization of the reasoning steps."""
        try:
            if not steps:
                return self._create_empty_chart()
            
            # Prepare data for visualization
            x_positions = list(range(len(steps)))
            y_positions = [0] * len(steps)  # All on same level for simplicity
            
            # Extract step information
            step_texts = []
            step_colors = []
            step_sizes = []
            
            for step in steps:
                content = step.get("content", "")
                step_type = step.get("type", "reasoning")
                
                # Truncate text for display
                display_text = self._truncate_text(content, max_length=30)
                step_texts.append(display_text)
                
                # Assign color based on step type
                color = self.step_colors.get(step_type, self.default_color)
                step_colors.append(color)
                
                # Size based on content length (within reasonable bounds)
                size = min(max(len(content) / 10 + 20, 25), 50)
                step_sizes.append(size)
            
            # Create the figure
            fig = go.Figure()
            
            # Add connecting lines between steps
            if len(steps) > 1:
                for i in range(len(steps) - 1):
                    fig.add_trace(go.Scatter(
                        x=[x_positions[i], x_positions[i + 1]],
                        y=[y_positions[i], y_positions[i + 1]],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Add step nodes
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=y_positions,
                mode='markers+text',
                marker=dict(
                    size=step_sizes,
                    color=step_colors,
                    line=dict(width=2, color='white')
                ),
                text=[f"Step {i+1}" for i in range(len(steps))],
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                hovertext=step_texts,
                hoverinfo='text',
                showlegend=False
            ))
            
            # Update layout
            fig.update_layout(
                title="Reasoning Flow",
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False
                ),
                plot_bgcolor='white',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating flowchart: {e}")
            return self._create_error_chart()
    
    def create_step_distribution(self, steps: List[Dict[str, str]]):
        """Create a pie chart showing the distribution of step types."""
        try:
            if not steps:
                return self._create_empty_chart()
            
            # Count step types
            step_counts = {}
            for step in steps:
                step_type = step.get("type", "reasoning")
                step_counts[step_type] = step_counts.get(step_type, 0) + 1
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(step_counts.keys()),
                values=list(step_counts.values()),
                hole=0.3,
                marker_colors=[self.step_colors.get(k, self.default_color) for k in step_counts.keys()]
            )])
            
            fig.update_layout(
                title="Step Type Distribution",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating step distribution: {e}")
            return self._create_error_chart()
    
    def create_step_timeline(self, steps: List[Dict[str, str]]):
        """Create a timeline visualization of the reasoning steps."""
        try:
            if not steps:
                return self._create_empty_chart()
            
            # Prepare timeline data
            step_numbers = list(range(1, len(steps) + 1))
            step_types = [step.get("type", "reasoning") for step in steps]
            step_contents = [self._truncate_text(step.get("content", ""), 50) for step in steps]
            
            # Create bar chart
            fig = go.Figure(data=[go.Bar(
                x=step_numbers,
                y=[1] * len(steps),  # All bars same height
                text=step_contents,
                textposition='inside',
                marker_color=[self.step_colors.get(t, self.default_color) for t in step_types],
                hovertext=[f"Step {i}: {content}" for i, content in enumerate(step_contents, 1)],
                hoverinfo='text'
            )])
            
            fig.update_layout(
                title="Reasoning Timeline",
                xaxis_title="Step Number",
                yaxis=dict(showticklabels=False),
                height=200,
                margin=dict(l=20, r=20, t=40, b=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating timeline: {e}")
            return self._create_error_chart()
    
    def _truncate_text(self, text: str, max_length: int = 50) -> str:
        """Truncate text to specified length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
    
    def _create_empty_chart(self):
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No steps to visualize",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig
    
    def _create_error_chart(self):
        """Create an error chart."""
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig
