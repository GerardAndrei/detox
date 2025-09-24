# DETOX

A modern, full-stack web application for cleaning and analyzing datasets. Built with React frontend and FastAPI backend, featuring glassmorphism UI design and powerful data processing capabilities.

## ✨ Features

- **Modern React Frontend** with glassmorphism UI design
- **FastAPI Backend** with RESTful endpoints
- **Drag & Drop File Upload** with real-time validation
- **Interactive Data Preview** with comprehensive statistics
- **Smart Data Cleaning** with customizable options:
  - Handle missing values automatically
  - Remove duplicate entries
  - Detect and analyze outliers (IQR & Z-Score methods)
- **Real-time Results** with detailed metrics and visualizations
- **Download Cleaned Data** as CSV files
- **Responsive Design** that works on all devices

## 🚀 Tech Stack

### Frontend
- **React 18** - Modern UI library
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations
- **React Dropzone** - Drag & drop file uploads
- **Lucide React** - Beautiful icons
- **React Hot Toast** - Toast notifications
- **Axios** - HTTP client

### Backend
- **FastAPI** - Modern Python web framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Uvicorn** - ASGI server
- **Python Multipart** - File upload handling

## 📁 Project Structure

```
detox/
│
├── frontend/                 # React frontend application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.js
│   │   │   ├── FileUpload.js
│   │   │   ├── DataPreview.js
│   │   │   ├── CleaningOptions.js
│   │   │   └── Results.js
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── backend/                  # FastAPI backend with performance optimizations
│   ├── main.py              # Enhanced backend with async processing
│   ├── requirements.txt
│   └── uploads/             # File upload directory
│
├── cleaner/                 # Data processing modules
│   ├── __init__.py
│   ├── profiling.py         # AI-powered data profiling
│   ├── cleaning.py          # Core cleaning algorithms
│   ├── outlier.py           # Outlier detection methods
│   ├── reporting.py         # Cleaning report generation
│   ├── optimized_processor.py  # Memory-efficient processing
│   ├── caching.py           # Multi-tier caching system
│   └── database.py          # Database operations
│
├── data/
│   └── sample.csv           # Sample dataset
│
└── README.md               # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v16 or higher)
- Python (v3.8 or higher)
- npm or yarn

### Backend Setup

1. **Navigate to the project root:**
```bash
cd "detox"
```

2. **Install Python dependencies:**
```bash
pip install -r backend/requirements.txt
```

3. **Start the FastAPI server:**
```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to the frontend directory:**
```bash
cd frontend
```

2. **Install Node.js dependencies:**
```bash
npm install
```

3. **Start the React development server:**
```bash
npm start
```

The application will open at `http://localhost:3000`

## 🎮 Usage

1. **Start both servers** (backend on :8000, frontend on :3000)

2. **Open your browser** and navigate to `http://localhost:3000`

3. **Upload a CSV file** using the drag & drop interface

4. **Review your data** in the preview section with detailed statistics

5. **Configure cleaning options:**
   - Toggle missing value handling
   - Enable/disable duplicate removal
   - Choose outlier detection method

6. **Clean your dataset** and review the results

7. **Download the cleaned data** as a CSV file

## 🔧 API Endpoints

- `POST /upload` - Upload and analyze CSV file
- `POST /clean/{file_id}` - Clean dataset with options
- `GET /download/{file_id}` - Download cleaned dataset
- `GET /profile/{file_id}` - Get detailed dataset profile
- `GET /` - Health check endpoint

## 🎨 Design Features

- **Glassmorphism UI** - Modern translucent design with backdrop blur
- **Smooth Animations** - Framer Motion powered transitions
- **Responsive Layout** - Works perfectly on desktop, tablet, and mobile
- **Interactive Elements** - Hover effects and micro-interactions
- **Real-time Feedback** - Toast notifications and loading states

## 🧪 Sample Data

A sample dataset (`data/sample.csv`) is included for testing. It contains employee information with intentional data quality issues to demonstrate the cleaning capabilities.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🚀 Future Enhancements

- [ ] User authentication and file management
- [ ] Advanced data visualization charts
- [ ] Machine learning-based data quality scoring
- [ ] Support for multiple file formats (Excel, JSON, etc.)
- [ ] Collaborative data cleaning workflows
- [ ] Data transformation pipelines
- [ ] Export to various formats (Parquet, JSON, etc.)

---

Built with ❤️ using React, FastAPI, and modern web technologies.