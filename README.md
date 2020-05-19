# Deploy a Malaria detection Model to Server production
this project tested, can be used in python 3.6 and 3.7.

## Requirement
- **gifsicle** you must setup gifsicle with manual.
  -  **Windows**: install as follows this [Guide](https://www.youtube.com/watch?v=5gdhQyP9Eog)
  -  **Linux** : install as follows: ```sudo apt-get install gifsicle```
<!-- install ffmpeg Follow [Guide](https://www.ffmpeg.org/download.html) -->

## Dependencies Installation
```python setup.py build_ext --inplace```<br>
```pip install -r requirements.txt```<br>
```conda config --add channels conda-forge```<br>
```conda install ffmpeg```

## Usage

### Change SERVER_IP
if you connect internet with LAN or Eternet, you should change ```interface = "eth"``` in ```app.py``` line 33 (```SERVER_IP```) or config manual by change SERVER_IP in else condition.

Can see more in ```src/get_ip.py```  for auto get ip.

### RUN
Just run this command:

```python app.py```

That's it! It's serving a saved Keras model to you via Flask. 

## Postman
you can use ```Malaria.postman_collection.json``` to API testing on postman. you must change URL.<br>
- **/GET getTest** : GET API testing<br>
- **/POST upload** : Upload video testing , you can change file path in Body option.<br>
- **/GET upload** :  get progress and result when finished.<br>