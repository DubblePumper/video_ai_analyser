# Video AI Analyser

An AI-powered video content analysis tool.

## Project Structure
`
VideoAIAnalyzer/
├── app/                                  # Main application directory
│   ├── __pycache__/                     # Python cache directory
│   ├── analyze_video.py                 # Main video analysis pipeline
│   ├── audio_analysis.py                # Audio content analysis module
│   ├── config.py                        # Configuration settings and parameters
│   ├── datasets/                        # Directory for training and reference data
│   │   └── final.csv                    # Final processed dataset
│   ├── detect.py                        # Core detection algorithms
│   ├── download_videos.py               # Video downloading utilities
│   ├── face_detection.py                # Face detection implementation
│   ├── getimageofpeopleindateset.py    # Dataset image extraction tool
│   ├── input_images/                    # Directory for input image files
│   ├── input_videos/                    # Directory for input video files
│   ├── models/                          # AI model storage directory
│   │   └── trained_model.h5            # Pre-trained AI model
│   ├── object_detection.py              # Object detection implementation
│   ├── output_results/                  # Analysis results output directory
│   ├── recognize_people.py              # Person recognition module
│   ├── saved_ai/                        # Saved AI model states directory
│   ├── scene_recognition.py             # Scene analysis implementation
│   ├── serve_images.py                  # Image serving utility
│   ├── temp/                            # Temporary file storage
│   ├── trainaionimageanddescofpeople.py # AI training script for person detection
│   ├── utils.py                         # Helper functions and utilities
│   └── verify_gpu.py                    # GPU availability check utility
├── docker-compose.yml                    # Docker compose configuration
├── Dockerfile                           # Docker container definition
├── LICENSE                              # Project license file
├── README.md                            # Project documentation (this file)
├── requirements.txt                      # Python package dependencies
└── .gitattributes                       # Git attributes configuration
`

## Key Components

### Core Analysis Modules
- `analyze_video.py`: Main entry point for video analysis
- `detect.py`: Core detection algorithms for content analysis
- `face_detection.py`: Specialized module for face detection
- `object_detection.py`: General object detection capabilities
- `scene_recognition.py`: Scene analysis and classification
- `audio_analysis.py`: Audio content analysis

### Data Management
- `datasets/`: Contains training data and reference materials
- `input_videos/`: Directory for processing video files
- `input_images/`: Directory for processing image files
- `output_results/`: Storage for analysis results
- `temp/`: Temporary processing files

### AI and Training
- `models/`: Contains trained AI models
- `saved_ai/`: Backup of AI model states
- `trainaionimageanddescofpeople.py`: Training pipeline for person detection

### Utilities
- `utils.py`: Common utility functions
- `config.py`: Application configuration
- `verify_gpu.py`: GPU verification tool
- `serve_images.py`: Image serving utility
- `download_videos.py`: Video download functionality

### Container Configuration
- `Dockerfile`: Container image definition
- `docker-compose.yml`: Container orchestration setup

## Requirements
See `requirements.txt` for full list of dependencies
