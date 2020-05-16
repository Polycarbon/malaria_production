# Deploy a Malaria detection Model to Production


## Requirement
- **gifsicle** you must setup gifsicle with manual.
  -  **Windows**: install as follows this [Guide](https://www.youtube.com/watch?v=5gdhQyP9Eog)
  -  **Linux** : install as follows: ```sudo apt-get install gifsicle```
<!-- install ffmpeg Follow [Guide](https://www.ffmpeg.org/download.html) -->

## Dependencies Installation
```python setup.py build_ext --inplace```<br>
```sudo pip install -r requirements.txt```<br>
```conda config --add channels conda-forge```<br>
```conda install ffmpeg```

## Usage

Once dependencies are installed, just run this to see it in your browser. 

```python app.py```

That's it! It's serving a saved Keras model to you via Flask. 