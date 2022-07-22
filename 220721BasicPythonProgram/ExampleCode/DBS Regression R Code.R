d = read.csv("C:/Users/Teoh Teik Teo/Dropbox/PC/Downloads/DBS_SingDollar (1).csv")
model = lm(d$DBS ~ d$SGD)
pred = predict(model)
err = d$DBS - pred
rmse = mean(err ^ 2) ^ 0.5

d$SGD2 = d$SGD ^ 2
model = lm(d$DBS ~ d$SGD + d$SGD2)
pred = predict(model)
err = d$DBS - pred
rmse1 = mean(err ^ 2) ^ 0.5

d$SGD3 = d$SGD ^ 3
model = lm(d$DBS ~ d$SGD + d$SGD2 + d$SGD3)
pred = predict(model)
err = d$DBS - pred
rmse2 = mean(err ^ 2) ^ 0.5

