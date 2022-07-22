from flask import Flask, request , render_template
import joblib

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        rates = float(request.form.get("rates"))
        print(rates)
        model_re = joblib.load("regression")
        r1 = model_re.predict([[rates]])
        print(r1)
        model_tree = joblib.load("tree")
        r2 = model_tree.predict([[rates]])
        print(r2)
        
        return (render_template("index.html", result_re = r1, result_tree = r2))
    else:
        return (render_template("index.html", result_re = "WAITTING", result_tree = "WAITTING"))
        
if __name__ == "__main__":
    app.run()