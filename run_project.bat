@echo off
echo ==========================================
echo     SPAM EMAIL CLASSIFIER
echo ==========================================
echo.

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Downloading and preparing real email data...
python download_dataset.py

echo.
echo Training Naive Bayes model...
python train.py

echo.
echo Starting GUI...
python gui.py

pause
