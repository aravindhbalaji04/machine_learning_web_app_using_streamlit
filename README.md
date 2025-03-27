# Machine Learning Classifier Explorer

This Streamlit web application allows users to explore different machine learning classifiers and determine which one performs the best on various datasets.

## Features
- Select different datasets (Iris, Breast Cancer, Wine)
- Choose from multiple classifiers (KNN, SVM, Random Forest)
- Tune classifier parameters using interactive sliders
- View dataset shape and class distribution
- Visualize the dataset using PCA (Principal Component Analysis)
- Display classifier accuracy

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Matplotlib
- NumPy

## Installation
To run this project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run main.py
```

## Usage
1. Open the Streamlit app in your browser.
2. Select a dataset from the sidebar.
3. Choose a classifier and adjust the parameters using the sliders.
4. View the dataset shape, number of classes, accuracy score, and PCA visualization.

## File Structure
```
├── main.py              # Streamlit app
├── requirements.txt     # Required dependencies
├── README.md            # Project documentation
```

## Requirements
Ensure you have the following installed:
- Python 3.x
- Streamlit
- Scikit-learn
- Matplotlib
- NumPy

## Contributing
Feel free to fork this repository and submit pull requests for improvements or new features.

## License
This project is open-source and available under the [MIT License](LICENSE).

---

**Note:** If you're deploying this on Streamlit Cloud, ensure that `requirements.txt` is present in your repository for proper package installation.
