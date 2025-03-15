### 6. run.sh (For Linux/Mac)

```bash
#!/bin/bash

# Create and activate virtual environment (if needed)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting 2D to 3D Video Converter..."
streamlit run app.py
