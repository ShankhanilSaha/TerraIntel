# TerraIntel Mobile Application

## Overview
A tool which takes terrain data from sattelite, drone shots performs analysis and gives optimized routes and plans powered with real-time tracking features for most optimal results.
## Features

- Interactive map view for real-time terrain analysis
- Tools for plotting attack and defense plans
- Import and visualize custom geospatial data (e.g., DEMs, satellite imagery)
- Organizational tools for missions, objectives, and strategies 
- Seamless user experience for mobile planning and collaboration

## Technology Stack

- **Android (Kotlin/JetpackCompose)**: Native development for performance and smooth UI
- **Google Earth Engine, USGS Earth Explorer, Bhuvan Open Data Archive, QGis**: For map rendering and data overlays
- **MongoDB**: For storing terrain data, past strategies.
- **Llama3.2, Djikstra**: For path analysis, strategy generation.

## Getting Started

### Prerequisites

- Android Studio (latest version recommended)
- Android device or emulator (API 21+)
- [Optional] Firebase account for backend services

### Setup Instructions

1. **Clone the repository**
```git clone https://github.com/ShankhanilSaha/WarPlanner.git```

2. **Checkout the mobile-application branch**
```git checkout mobile-application```
3. **Open in Android Studio**
- Select `File > Open`, browse to the project folder.
4. **Configure dependencies**
- Run Gradle sync to download all required dependencies.
- Add any necessary API keys (e.g., for Maps) in `local.properties` or environment variables.
5. **Run the app**
- Select a target device/emulator and click 'Run'.

