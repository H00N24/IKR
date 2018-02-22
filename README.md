Zavislosti:
  scikit-learn (pip3 install scikit-learn)
  scikit-image (pip3 install scikit-image)

Stiahnut a rozbalit trenovacie a testovacie data do priecinku k skriptom.

Spustenie:
  trenovaci mod:
    python3 classifiers.py test
    ./classifiers.py test
  eval mod:
    python3 classifiers.py
    ./classifiers.py

Vypise sa aktualna konfiguracia klasifikatora a cas trenovania

Pri trenovacom mode sa vygeneruje skore

Pri eval mode sckript generuje subory `image_[NAME].txt` kde [NAME] je meno
pouziteho klasifikatora: svc = Support Vector Machines
                         mlp = Multi-layer Perceptron
