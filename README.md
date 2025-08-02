# 🧠 ThoughtChain: Chain of Thought Reasoning Visualizer

A powerful Streamlit application that demonstrates **real Chain of Thought reasoning** using the Microsoft Phi-2 language model. This tool provides interactive visualizations of how AI models think through complex problems step-by-step.

## ✨ Features

- **Real AI Reasoning**: Uses actual Microsoft Phi-2 (2.7B parameters) model for genuine Chain of Thought generation
- **Interactive Visualizations**: Flowcharts, step distributions, and timeline views of reasoning processes
- **Smart Problem Detection**: Automatically categorizes problems (math, logic, riddles) and optimizes prompting
- **Hardware Optimization**: Automatic GPU/CPU detection with optional 4-bit quantization for memory efficiency
- **Configurable Generation**: Adjustable temperature, top-p, and response length parameters
- **Rich Analytics**: Generation time tracking, step classification, and performance metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- 8GB+ RAM (16GB+ recommended)
- GPU with 8GB+ VRAM (optional, but recommended for better performance)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ThoughtChain.git
cd ThoughtChain

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test installation (optional but recommended)
python test_installation.py
```

### Testing the Installation

Before running the full application, you can test if everything works:

```bash
# Run the comprehensive demo
python core/demo.py
```

This demo will:
- ✅ Load the Phi-2 model
- ✅ Test Chain of Thought generation
- ✅ Verify problem type detection
- ✅ Check performance metrics
- ✅ Test error handling

### Running the Application

```bash
# Start the Streamlit application
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 🔧 Configuration

### Model Settings

The application provides several configuration options in the sidebar:

- **4-bit Quantization**: Reduces memory usage by ~50% with minimal performance impact
- **Max Response Length**: Control the length of generated reasoning (100-1000 tokens)
- **Temperature**: Controls randomness (0.1-1.5, higher = more creative)
- **Top-p (Nucleus Sampling)**: Controls response diversity (0.1-1.0)

### Hardware Support

The application automatically detects and uses the best available hardware:

- **CUDA GPUs**: Full acceleration with optional quantization
- **Apple Silicon (M1/M2)**: MPS acceleration support
- **CPU**: Fallback option (slower but works on any system)

## 📖 Usage Guide

### 1. Initialize the Model

1. Open the application in your browser
2. In the sidebar, configure your preferred settings
3. Click "🚀 Initialize Phi-2 Model"
4. Wait for the model to load (may take 2-5 minutes on first run)

### 2. Generate Reasoning

1. Enter a problem in the text area, or use the example buttons
2. Click "🚀 Generate Chain of Thought"
3. Watch the real-time progress as the model generates reasoning
4. Explore the results in the three tabs:
   - **💭 Reasoning Steps**: Step-by-step breakdown with classification
   - **📊 Flow Visualization**: Interactive flowcharts and analytics
   - **🔧 Generation Info**: Technical details and performance metrics

### 3. Example Problems

The application includes built-in examples for different problem types:

**Math Problems:**
- Train speed and distance calculations
- Percentage and tax calculations
- Work rate problems
- Geometric calculations

**Logic Problems:**
- Height ordering problems
- Logical deduction
- Race position problems
- Logical fallacy identification

**Riddles:**
- Wordplay and metaphors
- Creative thinking challenges
- Pattern recognition

## 🧪 Testing & Development

### Demo Script

The `core/demo.py` script provides a comprehensive end-to-end test:

```bash
python core/demo.py
```

**What the demo tests:**
- Model loading and initialization
- Chain of Thought generation for different problem types
- Problem type detection accuracy
- Step parsing and classification
- Performance metrics and consistency
- Error handling and cleanup

### Installation Testing

Test your installation with:

```bash
python test_installation.py
```

This will verify:
- ✅ All dependencies are installed
- ✅ PyTorch and CUDA/MPS support
- ✅ Transformers library functionality
- ✅ Optional: Model loading test

## 🏗️ Architecture

### Core Components

```
ThoughtChain/
├── app.py                 # Main Streamlit application
├── core/
│   ├── model.py          # Real Phi-2 model management
│   ├── cot_generator.py  # Chain of Thought generation
│   ├── examples.py       # Example problem database
│   └── demo.py           # End-to-end testing script
├── visualization/
│   ├── step_display.py   # Step visualization components
│   └── flowchart.py      # Interactive flowchart generation
├── requirements.txt      # Dependencies
└── test_installation.py # Installation verification
```

### Model Management

The `ModelManager` class handles:
- Automatic model loading with Hugging Face Transformers
- Hardware detection and optimization
- Memory management and cleanup
- Quantization configuration

### Chain of Thought Generation

The `CoTGenerator` class provides:
- Problem type detection and optimized prompting
- Real model inference with configurable parameters
- Enhanced step parsing for natural language outputs
- Performance tracking and analytics

### Visualization System

- **StepVisualizer**: Renders reasoning steps with icons and color coding
- **FlowchartGenerator**: Creates interactive Plotly visualizations
- **Real-time Analytics**: Generation time, step distribution, and performance metrics

## 🔍 Technical Details

### Model Specifications

- **Model**: Microsoft Phi-2 (2.7B parameters)
- **Framework**: Hugging Face Transformers
- **Optimization**: Optional 4-bit quantization via BitsAndBytes
- **Hardware**: CUDA/MPS/CPU support with automatic detection

### Performance Characteristics

- **Memory Usage**: ~5GB with quantization, ~10GB without
- **Generation Speed**: 2-10 seconds depending on hardware and response length
- **Accuracy**: High-quality reasoning with step-by-step breakdowns

### Prompt Engineering

The application uses sophisticated prompt engineering:
- Problem-specific templates for math, logic, and riddles
- Chain of Thought prompting techniques
- Context-aware guidance for different problem types

## 🛠️ Development

### Local Development

1. **Set up development environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with development mode:**
   ```bash
   streamlit run app.py --server.port 8501 --server.address localhost
   ```

3. **Test changes:**
   ```bash
   python core/demo.py  # Test core functionality
   python test_installation.py  # Verify dependencies
   ```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the demo script
5. Add tests if applicable
6. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

**Model Loading Fails:**
- Ensure you have sufficient RAM (8GB+)
- Try enabling 4-bit quantization
- Check your internet connection for model download
- Run `python core/demo.py` to test model loading

**Slow Performance:**
- Enable GPU acceleration if available
- Reduce max response length
- Use quantization for memory efficiency
- Check hardware detection in the sidebar

**Memory Errors:**
- Enable 4-bit quantization
- Close other applications
- Reduce batch size or response length
- Use the cleanup button to free memory

**Demo Script Issues:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (requires 3.11+)
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`

### System Requirements

**Minimum:**
- Python 3.11+
- 8GB RAM
- 5GB free disk space

**Recommended:**
- Python 3.11+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 10GB+ free disk space

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Microsoft for the Phi-2 model
- Hugging Face for the Transformers library
- Streamlit for the web framework
- Plotly for interactive visualizations

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Run the demo script to verify functionality
- Review the technical documentation

---

**Happy reasoning! 🧠✨**