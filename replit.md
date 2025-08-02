# Overview

The Chain of Thought Reasoning Visualizer is a Streamlit application that demonstrates how language models think through complex problems step-by-step. The application loads local language models and generates structured reasoning chains for various types of problems including math, logic, and riddles. It provides interactive visualizations through flowcharts and step-by-step displays to help users understand the AI's thought process.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses Streamlit as the web framework, providing an interactive interface with a main content area and sidebar configuration panel. The UI is organized into modular visualization components that handle step-by-step reasoning displays and flowchart generation using Plotly for interactive graphics.

## Model Management
The system implements a centralized ModelManager class that handles loading and inference with local Hugging Face transformer models. It supports multiple lightweight models including DialoGPT variants and TinyLlama, with optional 4-bit quantization for memory efficiency. The architecture includes automatic device detection (CUDA/MPS/CPU) and Streamlit caching for model persistence.

## Chain of Thought Generation
The CoTGenerator component implements template-based prompting strategies with automatic problem type detection. It categorizes problems into math, logic, riddle, or general types and applies appropriate reasoning templates. The system parses model outputs into structured reasoning steps with type classification.

## Visualization System
The visualization layer consists of two main components:
- StepVisualizer: Renders reasoning steps in an organized UI format with icons, color coding, and expandable sections
- FlowchartGenerator: Creates interactive Plotly flowcharts showing the reasoning flow with node sizing and coloring based on step types

## Data Flow
Problems flow from user input through problem type detection, template selection, model inference, response parsing into structured steps, and finally visualization rendering. The system maintains state through Streamlit's session management for model persistence and UI consistency.

# External Dependencies

## Machine Learning Framework
- **Transformers (Hugging Face)**: Core library for loading and running pre-trained language models locally
- **PyTorch**: Underlying tensor computation framework with CUDA/MPS support
- **BitsAndBytesConfig**: Quantization library for memory-efficient model loading

## Web Framework
- **Streamlit**: Complete web application framework providing UI components, caching, and session management

## Visualization
- **Plotly**: Interactive plotting library used for generating flowchart visualizations with customizable styling and interactivity

## Pre-trained Models
- **microsoft/DialoGPT-small/medium**: Conversational AI models optimized for dialogue generation
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0**: Lightweight chat-optimized language model for resource-constrained environments