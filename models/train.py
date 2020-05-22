import xlearn as xl

fm_model = xl.create_fm()
fm_model.setTrain("./output_train.csv")


param = {
    'task': 'reg',
    'lr': 2,
    'lambda': 0.5,
    'fold': 5,
    'k': 2
}

#fm_model.cv(param)

fm_model.fit(param, "./model.out")

fm_model.setTest("./output_test2.csv")
fm_model.predict("./model.out", "./output.txt")
