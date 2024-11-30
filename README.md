# Heavy Vehicle Restriction Alert System  

## Description  
The **Heavy Vehicle Restriction Alert System** is a computer vision-based solution designed to monitor and enforce restrictions on heavy vehicles entering restricted zones. This system leverages advanced deep learning models and NVIDIA DeepStream SDK to analyze real-time video feeds from surveillance cameras and ensure compliance with regulations.  

The project is ideal for industries and urban areas where restricted access for heavy vehicles is crucial for safety, traffic management, or environmental protection.  

## Features  
- **Real-time Detection**: High-speed detection and classification of vehicles using optimized deep learning models like YOLO.  
- **Violation Alerts**: Automatically generates alerts when a restricted vehicle is detected, enabling timely action.  
- **Edge Deployment**: Designed to work on edge devices, including NVIDIA Jetson platforms, for low latency and efficient processing.  
- **Customizable Rules**: Configurable settings to define restricted zones, vehicle types, and alert criteria.  
- **Scalable Integration**: Can be integrated with barrier systems, monitoring dashboards, or centralized control systems.  

## Applications  
- **Oil and Gas Facilities**: To ensure restricted zones are free of unauthorized vehicles for safety and compliance.  
- **Urban Traffic Management**: Regulating heavy vehicles in city centers or designated restricted zones.  
- **Industrial Security**: Monitoring entry points in industrial areas for compliance and safety.  

## Technical Overview  
- **Frameworks**: YOLO for vehicle detection and NVIDIA DeepStream for optimized real-time inference.  
- **Platform**: Built to run on NVIDIA Jetson devices or systems with CUDA-enabled GPUs.  
- **Languages**: Python, C++ (optional for integration).  
- **Dependencies**: NVIDIA De epStream SDK, CUDA, cuDNN, and other required libraries.

## Tech stack
- **DeepStream**: Streaming Analytics toolkit for video, image, audio processing/understanding.
- **TensorRT**: TensorRT is a machine learning framework that is used to run machine learning inference on Nvidia hardware.
- **YOLOV5**: YOLO is a advanced computer vision algorithm that is used in object detection.
- **Jetson**: It is a SOM device manufactured by Nvidia for running tensor intensive application.

## Setup and Usage  
### Prerequisites  
- NVIDIA Jetson device or CUDA-enabled system.  
- NVIDIA Driver (Version 535.154.05 or later), CUDA 12.2, and DeepStream 6.3 installed.  

### Installation  
1. Clone the repository:  
   ```bash  
   git clone [https://github.com/your-repo/heavy-vehicle-alert-system.git](https://github.com/ShwethaN25/Heavy-Vehicle-Restriction-Alert)
   
2. Build
   ```bash 
   make 
4. Run
   ```bash 
   ./FirstApp file:///path-to-your-.mp4

