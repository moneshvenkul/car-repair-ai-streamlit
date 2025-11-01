Car Repair AI - Streamlit 🚗
Developed by Monesh Venkul
All rights reserved. This repository and project are the intellectual property of Monesh Venkul. Unauthorized copying or use is prohibited without explicit permission.

Overview
Car Repair AI-Streamlit is an advanced, AI-powered application designed to assist DIY car repairers. Leveraging state-of-the-art machine learning models for image recognition and question answering, the application enables users to:

Upload photos of car parts for instant identification

Ask technical questions about repairs and get step-by-step instructions

Retrieve YouTube tutorial links and car model-specific information

Search through uploaded vehicle manuals and resources

Built with Python, PyTorch, HuggingFace Transformers, Streamlit, and integrated with the Kaggle car parts dataset (40 classes), this project demonstrates robust end-to-end capabilities for intelligent automotive assistance.

Features
Car Part Identification: Upload an image, and the vision agent recognizes the exact car part (from 40 classes).

Fine-Tuned Language Assistant: Get natural-language answers for automotive Q&A, repair, and replacement procedures.

Car Model Display: Shows car model names in repair instructions whenever available.

YouTube Tutorial Integration: Links to relevant repair/replacement videos.

Retrieval-Augmented Generation (RAG): Searches PDF manuals and online resources for more reliable answers.

Persistent History: MongoDB Atlas integration for saving Q&A history.

Streamlit UI: Clean, responsive, and interactive front end suitable for technical and non-technical users.

Getting Started
Prerequisites
Python 3.9+

See requirements.txt or enhanced_requirements.txt

For GPU use: CUDA-compatible machine & torch with CUDA support

Kaggle account for dataset access

Installation
bash
git clone https://github.com/moneshvenkul/car-repair-ai-streamlit.git
cd car-repair-ai-streamlit

# Setup environment and install dependencies
python -m venv venv
source venv/bin/activate   # On Linux/macOS
venv\\Scripts\\activate    # On Windows
pip install -r enhanced_requirements.txt
Data Setup
Download Kaggle Car Parts Dataset (40 classes) and unzip into data/car_parts_dataset/ with train/, test/, and valid/ structure.

(Optional) Place PDF car manuals in the pdfs/ folder.

Ensure .streamlit/secrets.toml is configured with your MongoDB Atlas URI and (optionally) Kaggle/Youtube API keys.

Model Fine-Tuning
Vision Model:

bash
python fine_tune_vision_agent.py
Language Model:

bash
python fine_tune_language_agent.py
Running the App
bash
streamlit run enhanced_car_repair_app.py
Usage
Upload an image of your car part.

Confirm or adjust part detection.

Ask a repair/maintenance question.

Get instructions, car model info, and YouTube tutorial links.

Review previous sessions in sidebar/history.

Directory Structure
text
.
├── enhanced_car_repair_app.py
├── fine_tune_vision_agent.py
├── fine_tune_language_agent.py
├── enhanced_requirements.txt
├── data/
│   └── car_parts_dataset/
│       ├── train/
│       ├── test/
│       └── valid/
├── pdfs/
├── models/
│   ├── vision_agent_finetuned/
│   └── language_agent_finetuned/
└── .streamlit/
    └── secrets.toml
Project Status & Roadmap
 Vision agent fine-tuned on car parts dataset

 Language agent fine-tuned for car repair Q&A

 Car model detection, YouTube integration

 End-to-end Streamlit UI and MongoDB integration

 (Planned) Additional car brands, multilingual support, mobile-first UI

License
All rights reserved: Monesh Venkul (2025)
This project is not open source. Duplication, distribution, or use without permission is strictly prohibited.

Acknowledgments
Kaggle Car Parts Dataset

HuggingFace Transformers, PyTorch, Streamlit

Author: Monesh Venkul
Feel free to contact for research or commercial licensing.

This README helps your project stand out as professional, AI-driven, and protected. Replace or expand the demo section and contact info as you prefer!
