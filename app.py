from flask import Flask, render_template,request
import joblib
app=Flask(__name__)
model=joblib.load("model.pkl")
@app.route("/",methods=["GET","POST"])
def home():
    prediction = None
    if request.method=="POST":
        long_hair=float(request.form["long_hair"])
        forehead_width_cm=float(request.form["forehead_width_cm"])
        forehead_height_cm=float(request.form["forehead_height_cm"])
        nose_wide=float(request.form["nose_wide"])
        nose_long=float(request.form["nose_long"])
        lips_thin=float(request.form["lips_thin"])
        distance_nose_to_lip_long=float(request.form["distance_nose_to_lip_long"])
        data=[[long_hair,forehead_width_cm,forehead_height_cm,nose_wide, nose_long,lips_thin,distance_nose_to_lip_long]]
        result=model.predict(data)
        prediction = result[0]
    return render_template("index.html",prediction=prediction)
if __name__=="__main__":
    app.run(debug=True)