
import keras
import os
import re

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

regex = re.compile(r"\d(.\d){0,2}")

def load_models():
    return list(filter(lambda x: x.startswith("model_") and x.endswith(".keras"), os.listdir("models")))

def show_models():
    models = load_models()
    model_performances = {}

    for model_name in models:
        model = keras.models.load_model("models/" + model_name)
        loss, acc = model.evaluate(x=x_test, y=y_test, verbose=0)
        model_performances[model_name[6:-6]] = acc

    for key in sorted(model_performances.keys()):
        print(f"{key:{ max(map( lambda x: len(x), model_performances.keys())) }}: {model_performances[key]:%}")

def save_model(model, version):
    model.save(f"models/model_{version}.keras")

def ask_for_save(model):
    show_models()
    version = input("Version: ")
    models = list(filter(lambda x: x.startswith("model_") and x.endswith(".keras"), os.listdir("models")))
    if version == "":
        print("Not saving")
    elif regex.fullmatch(version) and f"model_{version}.keras" not in models:
        save_model(model, version)
        print("Saved model")
    else:
        print("Invalid version, retrying")
        ask_for_save(model)

if __name__ == "__main__":
    show_models()
