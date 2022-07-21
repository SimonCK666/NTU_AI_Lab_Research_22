print("hello r")

d = read.csv("E:\\NTUAILab\\220721BasicPythonProgram\\DBS_SingDollar.csv")

model = lm(d$DBS ~d$SGD, data = d)
print(model)

pred = predict(model, newdata = d)
print(pred)

rmse = mean((d$DBS - pred) ^ 2) ^ 0.5
print(rmse)

