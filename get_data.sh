pip install -q kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list
kaggle competitions download -c tabular-playground-series-jun-2022
mkdir data
unzip tabular-playground-series-jun-2022.zip -d data