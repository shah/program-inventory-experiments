import pandas as pd
from ludwig.api import LudwigModel

model = LudwigModel.load("./models-trained/legislative-text-only")
predictions = model.predict(data_csv="./predict-in-legislative.csv")
predictions.to_csv("./predict-out-legislative.csv")
model.close()
