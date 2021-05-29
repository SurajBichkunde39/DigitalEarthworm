import os
from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
from .crop_rec import CropRec
from .utils import plot_graph, generate_radom_string
from .clf_model import Model_clf


# env = jinja2.Environment()
# env.globals.update(zip=zip)

static_url_path = os.path.join(os.getcwd(), 'application/static')
app = Flask(__name__, static_url_path=static_url_path)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]iasdfffsd/'
app.jinja_env.filters['zip'] = zip


rec_model = CropRec()
rec_model.load_model()
clf_model = Model_clf()
clf_model.load_model()

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])


def allowed_file(filename):
    mask1 = "." in filename
    mask2 = filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    return mask1 and mask2


@app.route('/', methods=["GET", "POST"])
def hello_world():
    content = None
    if request.method == "POST":
        features = []
        # features.append('post method called')
        features.append(request.form.get('N'))
        features.append(request.form.get('P'))
        features.append(request.form.get('K'))
        features.append(request.form.get('Temperature'))
        features.append(request.form.get('Humidity'))
        features.append(request.form.get('PH'))
        features.append(request.form.get('Rainfall'))
        label, probability = rec_model.predict(features)
        # print(label, probability)
        path = plot_graph(probability)
        # print(path)
        feat_all = []
        for feat in features:
            feat_all.append(float(feat))
        feature_names = ['Nitrogen', 'phosphorus',
                         'potassium', 'Temperature',
                         'Humidity', 'PH', 'Rainfall']
        content = {
            "label": label[0],
            "plot_path": path,
            "probability": probability,
            "original_input": feat_all,
            "feature_names": feature_names
        }
        return render_template('index.html', content=content)
    return render_template('index.html', content=content)


@app.route('/clf', methods=["GET", "POST"])
def classification():
    content = None
    if request.method == 'POST':
        print("post method called")
        print(request.files)
        if "file" not in request.files:
            flash("No file part", 'error')
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("No selected file", 'warning')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            UPLOAD_FOLDER = os.path.join(os.getcwd(), "application/upload_dir/")
            middle_str = generate_radom_string()
            filename = filename.split('.')
            new_filename = filename[0]+middle_str+'.'+filename[1]
            final_filename = os.path.join(UPLOAD_FOLDER, new_filename)
            print("Final filename => ", final_filename)
            file.save(final_filename)
            # print(clf_model.predict(final_filename))
            content = {}
            label, plot_path, all_prob = clf_model.predict(final_filename)
            content['label'] = label
            content['plot_path'] = plot_path
            content['all_prob'] = all_prob
            content['final_filename'] = final_filename
            return render_template("classification.html", content=content)
        else:
            error_str = "Either file is not a png,jpg,JPEG or the fortmat is not right"
            flash(error_str, 'error')
            flash("Please try again", 'info')
            return redirect(request.url)
    return render_template("classification.html", content=content)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=80)
