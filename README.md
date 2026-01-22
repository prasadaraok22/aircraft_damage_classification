# aircraft_damage_classification


# Cretae virtual environment    
<ol class="linenums"><li class="L0"><code class="language-shell"><span class="pln">pip install virtualenv  # OR pip3 install virtualenv </span></code></li><li class="L1"><code class="language-shell"><span class="pln">virtualenv my_env </span><span class="com"># create a virtual environment named my_env</span></code></li><li class="L2"><code class="language-shell"><span class="pln">source my_env</span><span class="pun">/</span><span class="pln">bin</span><span class="pun">/</span><span class="pln">activate </span><span class="com"># activate my_env</span></code></li></ol>


# Install necessary packages
pip install -r requirements.txt

# Run the app : It will load VGG16 model and train using sample dataset and save the model
python main.py

# Test with sample image
python predict.py





