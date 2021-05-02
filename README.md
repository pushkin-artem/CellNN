# CellNN

Scriptek:
- main.py (ezt kell futtatni)
- config (EEG adatbázis definialása)
- ioprocess.py (EEG fájlok feldolgozása)
- feature_extraction.py (featurek kinyerése, jelenleg csak nyers adatokat továbbít)
- classifier.py (neurális hálók inicializalása)
- ai.py (neurális hálók definialása)

Futtatás és paraméterek valásztása:

- Main futtatása előtt ki kell csomágolni a BBCI.DE nevű zip fájlt és bele kell helyezni a projekt mappába.
- main.py álján megtalálható a program meghívása és ennek paraméterei:

if __name__ == '__main__':
    bci = BCISystem()
    bci.offline_processing(
        db_name=Databases.BBCI_DE,
        feature_type=FeatureType.RAW,
        classifier_type=ClassifierType.CNN) 
        
Jelenlegi prográm csak BBCI_DE, RAW db_name és feature_type paraméterekkel működik. A classifier_type lehet ClassifierType.CNN vagy ClassifierType.CellNN.

ClassifierType.CNN:
- konvoluciós neurális halózat (tesztelve lett)

ClassifierType.CellNN:
- celluláris neurális halózat (nem lett tesztelve cuda és NVIDIA driverek miatt, nem sikerült ezeket telepíteni)
